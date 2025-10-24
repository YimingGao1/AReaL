import torch
import torch.nn as nn
from areal.engine.ppo.actor import FSDPPPOActor
from areal.envs.glove_rotation.joint_model import QwenConnectorJoint

class UniPicActor(FSDPPPOActor):
    """自定义Actor，训练UniPic2的Qwen+Connector"""
    
    def __init__(self, config):
        # 先调用父类初始化
        super().__init__(config)
        
        # 加载联合模型
        self.joint_model = QwenConnectorJoint(
            qwen_path=config.qwen_model_path,
            connector_path=config.unipic_checkpoint_path
        )
        
        # 设置AReaL需要的属性
        self.model = self.joint_model
        self.config = self.joint_model.config
    
    def forward(self, batch):
        """前向传播"""
        # 从batch中提取输入
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        pixel_values = batch.get("pixel_values")
        image_grid_thw = batch.get("image_grid_thw")
        
        # 调用联合模型
        prompt_embeds, pooled_prompt_embeds = self.joint_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        
        return prompt_embeds, pooled_prompt_embeds
    
    def compute_logp(self, batch):
        """计算log概率"""
        # 暂时返回随机值，后面再完善
        batch_size = batch["input_ids"].shape[0]
        return torch.randn(batch_size, device=self.device)
    
    def get_input_embeddings(self):
        """获取输入embeddings"""
        return self.joint_model.get_input_embeddings()
    
    def get_output_embeddings(self):
        """获取输出embeddings"""
        return self.joint_model.get_output_embeddings()
    
    @property
    def device(self):
        """获取设备"""
        return self.joint_model.device
    
    @property
    def dtype(self):
        """获取数据类型"""
        return self.joint_model.dtype