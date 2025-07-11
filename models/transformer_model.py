import torch.nn as nn
from .base_model import BaseModel
import torch

class TransformerModel(BaseModel):
    def __init__(self, input_dim=768, num_heads=8, num_layers=4, num_classes=6, dropout=0.3):
        super().__init__(num_classes)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        self.input_proj = None  # Optional projection layer
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # If input_dim mismatches, project input to expected dimension
        if x.size(-1) != self.classifier.in_features:
            if self.input_proj is None:
                self.input_proj = nn.Linear(x.size(-1), self.classifier.in_features).to(x.device)
            x = self.input_proj(x)
        
        # Transformer需要添加位置编码
        # 这里使用简单的可学习位置编码
        if not hasattr(self, 'pos_encoder') or self.pos_encoder.size(2) != x.size(2) or self.pos_encoder.size(1) != x.size(1):
            self.pos_encoder = nn.Parameter(
                torch.zeros(1, x.size(1), x.size(2))
            ).to(x.device)
        
        x = x + self.pos_encoder
        
        # Transformer处理
        transformer_out = self.transformer_encoder(x)
        # transformer_out shape: (batch_size, seq_len, input_dim)
        
        # 取第一个时间步的分类token或平均池化
        pooled = transformer_out.mean(dim=1)
        pooled = self.dropout(pooled)
        
        logits = self.classifier(pooled)
        return logits