import torch
import torch.nn as nn

__all__ = ['MLP']


class MLP(nn.Module):
    
    def __init__(self, num_layers, num_feat, num_classes):
        super(MLP, self).__init__()
        layers = []       
        feat_list = [360] + [num_feat] * num_layers 
        
        for i in range(len(feat_list)-1):
            layers.append(nn.Linear(feat_list[i], feat_list[i+1]))        
            if i < len(feat_list) - 2:
                layers.append(nn.BatchNorm1d(feat_list[i+1]))
            layers.append(nn.ReLU())
            
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(feat_list[-1], num_classes)
        self.name = 'MLP'
    
    def forward(self, x, remove_last_layer=False):
        x = self.layers(x)
        if not remove_last_layer:
            x = self.fc(x)
        return x
            
     