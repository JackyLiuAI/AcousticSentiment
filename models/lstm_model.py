import torch.nn as nn
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__(num_classes)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,  # 输入特征维度 (768 for Wav2Vec2)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # 确保输入维度正确
        assert x.size(2) == self.lstm.input_size, \
            f"Input feature size {x.size(2)} does not match LSTM input size {self.lstm.input_size}"
        
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim*2)
        
        # 取最后一个时间步
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        
        logits = self.classifier(last_out)
        return logits