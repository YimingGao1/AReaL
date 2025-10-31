# test_image_editing.py
import torch
from PIL import Image
from areal.envs.glove_rotation.env import GloveRotationEnv
from areal.envs.glove_rotation.env_config import GloveRotationEnvConfig

# 加载环境
config = GloveRotationEnvConfig(
    glove_image_path="assets/glove.png",
    target_image_path="assets/glove_90deg.png",
    max_steps=1
)
env = GloveRotationEnv(config)

# 加载 UniPic2（这部分会在 workflow 中加载）
from areal.workflow.vision_image_editing_agent import VisionImageEditingWorkflow
from transformers import AutoProcessor, AutoTokenizer
from areal.api.cli_args import GenerationHyperparameters

gconfig = GenerationHyperparameters(n_samples=1, max_new_tokens=512)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

workflow = VisionImageEditingWorkflow(
    gconfig=gconfig,
    tokenizer=tokenizer,
    processor=processor,
    qwen_model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    unipic_checkpoint_path="Skywork/UniPic2-Metaquery-9B",
    max_turns=1,
    num_inference_steps=20,
    guidance_scale=3.5,
)

# 测试一个 episode
import asyncio
data = {"env_name": "glove_rotation", "env_config": config}
result = asyncio.run(workflow._run_one_episode(None, data, "test"))
print(f"Reward: {result[2]}")