import asyncio
import uuid
import torch
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.envs.utils.env_load_utils import load_env_from_registry
from areal.envs.glove_rotation.unipic_generator import UniPicImageGenerator
from tensordict import TensorDict

class GloveRotationWorkflow(RolloutWorkflow):
    def __init__(self, gconfig, tokenizer, processor, qwen_model_path, 
                 unipic_checkpoint_path, num_inference_steps=20, 
                 guidance_scale=3.5, dump_dir=None):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.qwen_model_path = qwen_model_path
        self.unipic_checkpoint_path = unipic_checkpoint_path
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.dump_dir = dump_dir
        
        # 延迟加载UniPic模型
        self._image_generator = None
    
    def _get_image_generator(self):
        """延迟加载UniPic模型"""
        if self._image_generator is None:
            self._image_generator = UniPicImageGenerator(
                qwen_path=self.qwen_model_path,
                unipic_checkpoint_path=self.unipic_checkpoint_path
            )
        return self._image_generator
    
    async def _run_one_episode(self, engine, data, rid):
        # 从环境获取初始拳套图像
        env, seed = load_env_from_registry(data)
        init_obs, _ = env.reset(seed=seed)

        # 构建简单的prompt
        messages = [
            {"role": "system", "content": "You are a glove rotation assistant."},
            {"role": "user", "content": "Rotate this glove 90 degrees: <image>"}
        ]
        
        # 处理图像和文本
        images = init_obs["multi_modal_data"]["<image>"]
        text = self.processor.tokenizer.apply_chat_template(
            tokenize=False, add_generation_prompt=True, conversation=messages
        )
        
        processed_input = self.processor(
            images=images,
            text=text,
            padding=False,
            return_tensors="pt",
        )
        
        input_ids = processed_input["input_ids"].tolist()[0]
        
        # 生成固定的回复
        fixed_response = "rotate 90 degrees"
        response_tokens = self.tokenizer.encode(fixed_response, add_special_tokens=False)
        
        # 使用UniPic2生成旋转后的图像
        generator = self._get_image_generator()
        generated_image = generator.generate_edited_image(
            input_image=images[0],
            prompt=fixed_response,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        )
        
        # 计算MSE奖励
        reward = env._compute_mse_reward(generated_image)
        
        # 构建完整的token序列
        full_seq = input_ids + response_tokens
        seq_len = len(full_seq)

        # 构建logprobs（固定回复的logprobs设为0）
        logprobs = [0.0] * len(input_ids) + [0.0] * len(response_tokens)

        # 构建loss_mask（只训练回复部分）
        loss_mask = [0] * len(input_ids) + [1] * len(response_tokens)

        # 构建versions（设为-1）
        versions = [-1] * seq_len

        # 构建返回结果
        res = {
            "input_ids": torch.tensor(full_seq, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.long).unsqueeze(0),
            "rewards": torch.tensor([reward]),
        }

        # 添加多模态输入
        if "pixel_values" in processed_input:
            res["multi_modal_input"] = [{
                "pixel_values": processed_input["pixel_values"],
                "image_grid_thw": processed_input.get("image_grid_thw", None)
            }]
        
        return TensorDict(res, batch_size=[1])
    
    async def arun_episode(self, engine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        return concat_padded_tensors(results)