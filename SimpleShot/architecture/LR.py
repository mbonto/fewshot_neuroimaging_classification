import torch
import torch.nn as nn

__all__ = ['LR']


class LR(nn.Module):
    
    def __init__(self, num_classes):
        super(LR, self).__init__()
        self.fc = nn.Linear(360, num_classes)
        self.name = 'LR'
    
    def forward(self, x, remove_last_layer=False):
        if not remove_last_layer:
            x = self.fc(x)
        return x
            
     