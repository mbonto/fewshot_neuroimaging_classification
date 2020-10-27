import torch
import torch.nn as nn

__all__ = ['Conv1d']


class Conv1d(nn.Module):
    
    def __init__(self, num_layers, num_feat, num_classes, kernel=1):
        super(Conv1d, self).__init__()
        layers = []
        layers.append(nn.Conv1d(1, num_feat, kernel_size=kernel, stride=1, padding=0, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, num_layers):
            layers.append(nn.Conv1d(num_feat, num_feat, kernel_size=kernel, stride=1, padding=0, bias=True))
            layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(360, num_classes)
        self.name = 'Conv1d'
    
    def forward(self, x, remove_last_layer=False):
        x = self.layers(x)
        # print(x.shape, 'n_batch, n_channel, 360')
        # Average pooling by "pixels".
        x = torch.mean(x, dim=1)
        # print(x.shape, 'n_batch, 1, 360')
        x = torch.squeeze(x, dim=1)
        if not remove_last_layer:
            x = self.fc(x)
        return x