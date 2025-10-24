import os
from PIL import Image
import numpy as np
from areal.envs.registry import env_class
from areal.envs.glove_rotation.env_config import GloveRotationEnvConfig

@env_class("glove_rotation")
class GloveRotationEnv:
    def __init__(self, config: GloveRotationEnvConfig):
        self.config = config
        self.step_count = 0
        self.glove_image = None
        self.target_image = None
        self.total_reward = 0
        
        # 只加载图片，不加载模型
        self._load_images()

    def _load_images(self):
        """加载图片"""
        # 加载初始拳套图片
        if os.path.exists(self.config.glove_image_path):
            self.glove_image = Image.open(self.config.glove_image_path).convert("RGB")
        else:
            self.glove_image = Image.new("RGB", (100, 100), color="red")
        
        # 加载目标图片
        if os.path.exists(self.config.target_image_path):
            self.target_image = Image.open(self.config.target_image_path).convert("RGB")
        else:
            self.target_image = Image.new("RGB", (100, 100), color="blue")

    def reset(self, seed=None):
        """重置环境"""
        self.step_count = 0
        self.total_reward = 0
        self._load_images()
        return self._render(), {}

    def step(self, action_str: str):
        self.step_count += 1
        
        # 检查结束条件
        done = self.step_count >= self.config.max_steps
        
        return self._render(), 0.0, done, {}

    def _compute_mse_reward(self, generated_image):
        """计算MSE奖励"""
        gen_img = generated_image.resize(self.target_image.size)
        gen_pixels = np.array(gen_img).astype(np.float32)
        target_pixels = np.array(self.target_image).astype(np.float32)
        mse = np.mean((gen_pixels - target_pixels) ** 2)
        reward = 1.0 / (1.0 + mse)
        return reward

    def _render(self):
        """渲染状态"""
        return {
            "obs_str": f"Step: {self.step_count}, Action: rotate 90 degrees",
            "multi_modal_data": {
                "<image>": [self.glove_image]
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