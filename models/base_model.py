import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, x):
        raise NotImplementedError
        
    def get_optimizer(self, lr=1e-3, weight_decay=1e-4):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))