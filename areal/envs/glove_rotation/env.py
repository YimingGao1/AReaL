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
        self.current_image = None
        self.target_image = None
        self.total_reward = 0
        
        self._load_images()

    def _load_images(self):
        """加载初始图片和目标图片"""
        # 加载初始拳套图片
        if os.path.exists(self.config.glove_image_path):
            self.current_image = Image.open(self.config.glove_image_path).convert("RGB")
        else:
            # 如果没有图片，创建一个占位符
            self.current_image = Image.new("RGB", (512, 512), color="red")
        
        # 加载目标图片（旋转90度后的拳套）
        if os.path.exists(self.config.target_image_path):
            self.target_image = Image.open(self.config.target_image_path).convert("RGB")
        else:
            # 如果没有目标图片，创建一个占位符
            self.target_image = Image.new("RGB", (512, 512), color="blue")

    def reset(self, seed=None):
        """重置环境"""
        self.step_count = 0
        self.total_reward = 0
        self._load_images()
        return self._render(), {}

    def step(self, edited_image: Image.Image):
        """
        接收编辑后的图片，计算奖励
        
        Args:
            edited_image: UniPic2 编辑后的图片
            
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1
        
        # 更新当前图片
        self.current_image = edited_image
        
        # 计算与目标图片的 MSE 奖励
        reward = self._compute_mse_reward(edited_image)
        self.total_reward += reward
        
        # 检查是否结束
        done = self.step_count >= self.config.max_steps
        
        info = {
            "step": self.step_count,
            "mse_reward": reward,
            "total_reward": self.total_reward
        }
        
        return self._render(), reward, done, info

    def _compute_mse_reward(self, generated_image: Image.Image) -> float:
        """
        计算生成图片与目标图片的 MSE 奖励
        MSE 越小，奖励越高
        """
        # 确保尺寸一致
        gen_img = generated_image.resize(self.target_image.size)
        
        # 转换为 numpy 数组
        gen_pixels = np.array(gen_img).astype(np.float32) / 255.0
        target_pixels = np.array(self.target_image).astype(np.float32) / 255.0
        
        # 计算 MSE
        mse = np.mean((gen_pixels - target_pixels) ** 2)
        
        # 转换为奖励：MSE 越小，奖励越高
        # 使用负对数形式，使奖励在 [0, +∞) 范围内
        reward = 1.0 / (1.0 + mse)
        
        return float(reward)

    def _render(self):
        """返回当前观察"""
        obs_str = f"Please rotate the glove 90 degrees clockwise. Current step: {self.step_count}/{self.config.max_steps}"
        
        return {
            "obs_str": obs_str,
            "multi_modal_data": {
                "<image>": [self.current_image]
            }
        }

    def get_system_prompt(self):
        """返回系统提示"""
        return "You are an image editing assistant. Your task is to rotate the glove image to match the target orientation."

    def close(self):
        """关闭环境，释放资源"""
        if self.current_image:
            self.current_image.close()
            self.current_image = None
        if self.target_image:
            self.target_image.close()
            self.target_image = None