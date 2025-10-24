from dataclasses import dataclass
from areal.envs.registry import config_class

@config_class("glove_rotation")
@dataclass
class GloveRotationEnvConfig:
    """拳套旋转环境配置"""
    
    # 图片路径
    glove_image_path: str = "assets/glove.png"        # 初始拳套图片
    target_image_path: str = "assets/glove_90deg.png"  # 目标图片（旋转90度后）
    
    # 环境参数
    max_steps: int = 5
    render_mode: str = "vision"
    image_placeholder: str = "<image>"
    
    # 模型路径
    qwen_model_path: str = "/path/to/Qwen2.5-VL-7B-Instruct"
    unipic_checkpoint_path: str = "/path/to/unipic2_checkpoint"
    
    # 生成参数
    num_inference_steps: int = 20
    guidance_scale: float = 3.5
    image_size: int = 512