import asyncio
import os
import uuid
from typing import Any, Dict, List

import torch
from PIL import Image
from tensordict import TensorDict
from transformers import (
    AutoProcessor, 
    PreTrainedTokenizerFast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor
)
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.dataset.clevr_count_70k import convert_image
from areal.envs.utils.env_load_utils import load_env_from_registry
from areal.utils.data import concat_padded_tensors
from realhf.base import logging

# 导入 UniPic2 组件
import sys
unipic_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../UniPic/UniPic-2"))
sys.path.append(unipic_path)
from unipicv2.pipeline_stable_diffusion_3_kontext import StableDiffusion3KontextPipeline
from unipicv2.transformer_sd3_kontext import SD3Transformer2DKontextModel
from unipicv2.stable_diffusion_3_conditioner import StableDiffusion3Conditioner
        

logger = logging.getLogger("Vision Image Editing Agent workflow")


def fix_longer_edge(x: Image.Image, image_size: int, factor: int = 32) -> Image.Image:
    """调整图片尺寸保持长宽比"""
    w, h = x.size
    if w >= h:
        target_w = image_size
        target_h = h * (target_w / w)
        target_h = round(target_h / factor) * factor
    else:
        target_h = image_size
        target_w = w * (target_h / h)
        target_w = round(target_w / factor) * factor
    return x.resize(size=(int(target_w), int(target_h)))


class VisionImageEditingWorkflow(RolloutWorkflow):
    """
    基于 UniPic2 的图片编辑 RL workflow
    
    流程：
    1. 环境提供当前图片 + 目标描述
    2. LLM (Qwen2.5-VL) 理解图片和任务，生成隐藏层
    3. Conditioner 将隐藏层转换为 SD3 的 prompt embeddings
    4. SD3 Pipeline 编辑图片
    5. 环境计算编辑后图片与目标的 MSE 奖励
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        processor: AutoProcessor,
        # UniPic2 模型路径
        qwen_model_path: str,
        unipic_checkpoint_path: str,
        # 生成参数
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        image_size: int = 512,
        seed: int = 42,
        max_turns: int = 1,  # 暂时先用一轮的，后面再改成多轮的
        image_placeholder: str = "<image>",
        dump_dir: str | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.qwen_model_path = qwen_model_path
        self.unipic_checkpoint_path = unipic_checkpoint_path
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.seed = seed
        self.max_turns = max_turns
        self.image_placeholder = image_placeholder
        self.dump_dir = dump_dir
        
        # 延迟加载模型（在第一次使用时加载）
        self._lmm = None
        self._processor_qwen = None
        self._conditioner = None
        self._pipeline = None
        
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    def _load_models(self):
        """延迟加载 UniPic2 模型组件"""
        if self._lmm is not None:
            return  # 已加载
        
        logger.info("Loading UniPic2 models...")
        
       
        # 1. 加载 Qwen2.5-VL
        logger.info(f"Loading Qwen2.5-VL from {self.qwen_model_path}")
        self._lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).cuda()
        
        self._processor_qwen = Qwen2_5_VLProcessor.from_pretrained(self.qwen_model_path)
        # 移除默认的 system prompt
        self._processor_qwen.chat_template = self._processor_qwen.chat_template.replace(
            "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}",
            ""
        )
        
        # 2. 加载 Transformer
        logger.info("Loading SD3 Transformer...")
        transformer = SD3Transformer2DKontextModel.from_pretrained(
            self.unipic_checkpoint_path, 
            subfolder="transformer", 
            torch_dtype=torch.bfloat16
        ).cuda()
        
        # 3. 加载 VAE
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            self.unipic_checkpoint_path, 
            subfolder="vae", 
            torch_dtype=torch.bfloat16
        ).cuda()
        
        # 4. 加载 Conditioner
        logger.info("Loading Conditioner...")
        self._conditioner = StableDiffusion3Conditioner.from_pretrained(
            self.unipic_checkpoint_path, 
            subfolder="conditioner", 
            torch_dtype=torch.bfloat16
        ).cuda()
        
        # 5. 加载 Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.unipic_checkpoint_path, 
            subfolder="scheduler"
        )
        
        # 6. 创建 Pipeline（不使用 text encoders，因为用 Qwen 替代）
        logger.info("Creating SD3 Kontext Pipeline...")
        self._pipeline = StableDiffusion3KontextPipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            text_encoder_3=None,
            tokenizer_3=None,
            scheduler=scheduler
        )
        
        logger.info("✅ UniPic2 models loaded successfully!")

    async def _run_one_episode(
        self, engine: InferenceEngine, data: dict, rid: str
    ) -> TensorDict:
        """运行一个 episode"""
        
        # 加载模型（如果还未加载）
        self._load_models()
        
        # 加载环境
        env, seed = load_env_from_registry(data)
        
        try:
            # 重置环境
            init_obs, _ = env.reset(seed=seed)
            sys_prompt = env.get_system_prompt()
            
            # 获取当前图片
            current_image = init_obs["multi_modal_data"]["<image>"][0]
            if not isinstance(current_image, Image.Image):
                current_image = convert_image(current_image)
            
            # 调整图片尺寸
            current_image = fix_longer_edge(current_image, self.image_size)
            
            # ============ 准备输入 ============
            # 构建 prompt（系统提示 + 用户任务描述）
            prompt_text = init_obs.get("obs_str", "Rotate this glove 90 degrees")
            negative_prompt = init_obs.get("negative_prompt", 
                "blurry, low quality, distorted, deformed")
            
            messages = [
                [{"role": "user", "content": [
                    {"type": "image", "image": current_image},
                    {"type": "text", "text": prompt_text}
                ]}],
                [{"role": "user", "content": [
                    {"type": "image", "image": current_image},
                    {"type": "text", "text": negative_prompt}
                ]}]
            ]
            
            # 应用 chat template
            texts = [
                self._processor_qwen.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                ) 
                for msg in messages
            ]
            
            # 处理输入
            min_pixels = max_pixels = int(current_image.height * 28 / 32 * current_image.width * 28 / 32)
            inputs = self._processor_qwen(
                text=texts,
                images=[current_image] * 2,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                videos=None,
                padding=True,
                return_tensors="pt"
            ).to("cuda")
            
            # ============ 获取隐藏层 ============
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            pixel_values = inputs.pixel_values
            image_grid_thw = inputs.image_grid_thw
            
            # 添加 meta queries 占位符
            num_queries = self._conditioner.config.num_queries
            input_ids = torch.cat([
                input_ids, 
                input_ids.new_zeros(2, num_queries)
            ], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                attention_mask.new_ones(2, num_queries)
            ], dim=1)
            
            # 构建 inputs_embeds
            inputs_embeds = self._lmm.get_input_embeddings()(input_ids)
            inputs_embeds[:, -num_queries:] = self._conditioner.meta_queries[None].expand(2, -1, -1)
            
            # 处理图片 embeddings
            image_embeds = self._lmm.visual(pixel_values, grid_thw=image_grid_thw)
            image_token_id = self._processor_qwen.tokenizer.convert_tokens_to_ids('<|image_pad|>')
            inputs_embeds[input_ids == image_token_id] = image_embeds
            
            # 前向传播获取隐藏层
            self._lmm.model.rope_deltas = None
            with torch.no_grad():
                outputs = self._lmm.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    use_cache=False
                )
            
            # 提取最后 num_queries 个 token 的隐藏状态
            hidden_states = outputs.last_hidden_state[:, -num_queries:]
            
            # ============ 通过 Conditioner 得到 prompt embeds ============
            prompt_embeds, pooled_prompt_embeds = self._conditioner(hidden_states)
            
            # ============ 使用 SD3 Pipeline 编辑图片 ============
            logger.info(f"Editing image with prompt: {prompt_text}")
            edited_image = self._pipeline(
                image=current_image,
                prompt_embeds=prompt_embeds[:1],
                pooled_prompt_embeds=pooled_prompt_embeds[:1],
                negative_prompt_embeds=prompt_embeds[1:],
                negative_pooled_prompt_embeds=pooled_prompt_embeds[1:],
                height=current_image.height,
                width=current_image.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device=self._pipeline.transformer.device).manual_seed(self.seed)
            ).images[0]
            
            # ============ 将编辑后的图片传给环境计算奖励 ============
            next_obs, reward, done, info = env.step(edited_image)
            
            logger.info(f"✅ Episode completed! Reward: {reward:.4f}")
            
           
            
            # 为了兼容现有框架，我们构造一个虚拟的 token 序列
            dummy_input_ids = inputs.input_ids[0].cpu().tolist()
            seq_len = len(dummy_input_ids)
            
            res = {
                "input_ids": torch.tensor(dummy_input_ids, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
                "loss_mask": torch.zeros(1, seq_len, dtype=torch.long),  # 不训练 token 生成
                "logprobs": torch.zeros(1, seq_len, dtype=torch.float32),
                "versions": torch.full((1, seq_len), -1, dtype=torch.long),
                "rewards": torch.tensor([float(reward)]),
            }
            
            # 添加多模态输入
            multi_modal_input = {
                "pixel_values": pixel_values[:1],
                "image_grid_thw": image_grid_thw[:1]
            }
            res["multi_modal_input"] = [multi_modal_input]
            
            return (
                TensorDict(res, batch_size=[1]),
                prompt_text,
                reward,
                seq_len
            )
            
        finally:
            try:
                env.close()
            except Exception:
                pass


    async def arun_episode(self, engine: InferenceEngine, data: dict):
        """
        Public API to run one episode with potentially multiple samples.
        """
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        # Optional dump to disk
        if self.dump_dir is not None:
            version = engine.get_version()
            os.makedirs(os.path.join(self.dump_dir, str(version)), exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex
            with open(
                os.path.join(self.dump_dir, str(version), f"{qid}.txt"), "a"
            ) as f:
                n_samples = self.gconfig.n_samples
                for i, (_, prompt, reward, _) in enumerate(results):
                    f.write(f"n_samples: {self.gconfig.n_samples}, Sample {i+1}: prompt='{prompt}', reward={reward:.4f}\n")  


        td_list = [res[0] for res in results]
        return concat_padded_tensors(td_list)

    

   