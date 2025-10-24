import os
import numpy as np
from PIL import Image
from areal.envs.registry import env_class
from areal.envs.glove_rotation.env_config import GloveRotationEnvConfig
from areal.envs.glove_rotation.unipic_generator import UniPicImageGenerator

@env_class("glove_rotation")
class GloveRotationEnv:
    """拳套旋转环境"""
    
    def __init__(self, config: GloveRotationEnvConfig):
        self.config = config
        self.step_count = 0
        self.glove_image = None
        self.target_image = None
        self.total_reward = 0
        
        # 加载图片
        self._load_images()
        
        # 初始化联合模型
        self.image_generator = UniPicImageGenerator(
        qwen_path=self.config.qwen_model_path,
        unipic_checkpoint_path=self.config.unipic_checkpoint_path
    )

    
    def _load_images(self):
        """加载图片"""
        # 加载初始拳套图片
        if os.path.exists(self.config.glove_image_path):
            self.glove_image = Image.open(self.config.glove_image_path).convert("RGB")
        else:
            # 创建测试图片
            self.glove_image = Image.new("RGB", (100, 100), color="red")
        
        # 加载目标图片
        if os.path.exists(self.config.target_image_path):
            self.target_image = Image.open(self.config.target_image_path).convert("RGB")
        else:
            # 创建测试图片
            self.target_image = Image.new("RGB", (100, 100), color="blue")
    
    def reset(self, seed=None):
        """重置环境"""
        self.step_count = 0
        self.total_reward = 0
        if self.glove_image:
            self.glove_image.close()
        if self.target_image:
            self.target_image.close()
    
        # 重新加载
        self._load_images()
        return self._render(), {}  # 这里需要返回一个字典，包含obs_str和multi_modal_data
    
    def step(self, action_str: str):
        """执行一步"""
        # 固定动作："rotate 90 degrees"
        prompt = "rotate 90 degrees"
        
        edited_image = self.image_generator.generate_edited_image(
        input_image=self.glove_image,
        prompt=prompt,  # 比如 "rotate 90 degrees"
        num_inference_steps=self.config.num_inference_steps,
        guidance_scale=self.config.guidance_scale
        )
        
        self.glove_image = edited_image
        # 暂时用随机奖励，后面再集成UniPic2
        reward = np.random.uniform(-1, 1)
        
        self.step_count += 1
        self.total_reward += reward
        
        # 检查结束条件
        done = self.step_count >= self.config.max_steps
        
        return self._render(), reward, done, {}  #没有额外信息，只有拳头旋转90度，不需要info
    
    def _render(self):
        """渲染状态"""
        return {
            "obs_str": f"Step: {self.step_count}, Action: rotate 90 degrees",
            "multi_modal_data": {
                "<images>": [self.glove_image]
            }
        }
    
    def get_system_prompt(self):
        return "You are a glove rotation assistant. Rotate the glove to the target angle."
    
    def close(self):
        if self.glove_image:
            self.glove_image.close()
            self.glove_image = None
        if self.target_image:
            self.target_image.close()
            self.target_image = None