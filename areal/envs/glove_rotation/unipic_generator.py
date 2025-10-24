import torch
import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusion3KontextPipeline, AutoencoderKL, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from unipicv2.stable_diffusion_3_conditioner import StableDiffusion3Conditioner

class UniPicImageGenerator(nn.Module):
    """简化的UniPic图片生成器"""
    
    def __init__(self, qwen_path, unipic_checkpoint_path):
        super().__init__()
        print("Loading Qwen model...")
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).cuda()
        
        print("Loading UniPic2 conditioner...")
        self.conditioner = StableDiffusion3Conditioner.from_pretrained(
            unipic_checkpoint_path, 
            subfolder="conditioner",
            torch_dtype=torch.bfloat16
        ).cuda()
        
        print("Loading UniPic2 pipeline...")
        # 加载完整的UniPic2 pipeline
        self.pipeline = self._load_pipeline(unipic_checkpoint_path)
        
        print("UniPic image generator loaded successfully!")
    
    def _load_pipeline(self, checkpoint_path):
        """加载UniPic2 pipeline"""
        # 加载transformer
        transformer = StableDiffusion3KontextTransformer.from_pretrained(
            checkpoint_path, subfolder="transformer", torch_dtype=torch.bfloat16
        ).cuda()
        
        # 加载VAE
        vae = AutoencoderKL.from_pretrained(
            checkpoint_path, subfolder="vae", torch_dtype=torch.bfloat16
        ).cuda()
        
        # 加载text encoders
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            checkpoint_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer")
        
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            checkpoint_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer_2 = CLIPTokenizer.from_pretrained(checkpoint_path, subfolder="tokenizer_2")
        
        text_encoder_3 = T5EncoderModel.from_pretrained(
            checkpoint_path, subfolder="text_encoder_3", torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer_3 = T5TokenizerFast.from_pretrained(checkpoint_path, subfolder="tokenizer_3")
        
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            checkpoint_path, subfolder="scheduler"
        )
        
        # 创建pipeline
        pipeline = StableDiffusion3KontextPipeline(
            transformer=transformer, vae=vae,
            text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3, tokenizer_3=tokenizer_3,
            scheduler=scheduler
        )
        
        return pipeline
    
    def generate_edited_image(self, input_image, prompt, num_inference_steps=20, guidance_scale=3.5, seed=42):
        """生成编辑后的图片"""
        # 处理输入图片
        if isinstance(input_image, str):
            image = Image.open(input_image)
        else:
            image = input_image
        
        # 使用pipeline生成图片
        generator = torch.Generator(device=self.pipeline.transformer.device).manual_seed(seed)
        
        edited_image = self.pipeline(
            image=image,
            prompt=prompt,
            height=image.height, 
            width=image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return edited_image
    
    @property
    def device(self):
        return next(self.parameters()).device