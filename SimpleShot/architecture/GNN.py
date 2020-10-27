# Code inspired from the implementation of SGC in the original github repository https://github.com/Tiiiger/SGC.
import torch
import torch.nn as nn
from scipy import io as sio

__all__ = ['GNN']


# Load structural connectivity matrix.
# The connections are computed between the 360 regions defined in glasser. They are established from 
# structural MRI (T1, DWI). Self-connections are set to zero.
# DWI: Tractography looking at the strength of the white fibers between regions, averaged on 56 subjects.
### TO UPDATE
connectivity_matrix_path = '/home/brain/Myriam/fMRI_transfer/github/dataset/SC_avg56.mat'
connectivity = sio.loadmat(connectivity_matrix_path)['SC_avg56']


def compute_laplacian(adj_matrix, threshold=True):
    adj_matrix = torch.from_numpy(adj_matrix)
    # If threshold, we delete the connections whose strength is smaller than 1.
    if threshold:
        mask = (adj_matrix > 1).type(torch.float)
        adj_matrix = adj_matrix * mask
    # Maximal connection strength.
    maxi = torch.max(adj_matrix)
    # Add the identity matrix to the adjacency matrix.
    adj_matrix += torch.eye(adj_matrix.shape[0]) * maxi
    # Compute the degree matrix by summing the columns.
    degree = torch.sum(adj_matrix, dim=1)
    degree = torch.pow(degree, -1)
    degree = torch.diag(degree)
    # Compute the Laplacian.
    laplacian = torch.matmul(degree, adj_matrix)
    return laplacian.type(torch.float)


def propagation(features, graph, k=1):
    """
    Return the features propagated k times on the graph.
    
    Params:
        features -- tensor of size (batch, num_features).
        graph -- adjacency matrix of size (num_features, num_features).
        k -- number of times the features are propagated, integer.
    """
    features = torch.unsqueeze(features, dim=1)
    for _ in range(k):
        features = torch.matmul(features, graph)
    return torch.squeeze(features, dim=1)


class GNN(nn.Module):
    """
    Implementation of a Simplified Graph Convolution (propagation of the features of the graph on the graph, repeated k times) followed by a fully-connected layer. 
    """
    def __init__(self, num_classes, k, num_layers, num_feat):
        super(GNN, self).__init__()
        self.name = 'GNN'
        self.graph = compute_laplacian(connectivity).cuda() ##.cpu()
        self.k = k
        
        layers = []
        layers.append(nn.Linear(360, num_feat))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, num_layers):
            layers.append(nn.Linear(num_feat, num_feat))
            layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_feat, num_classes)

    def forward(self, x, remove_last_layer=False):
        x = propagation(x, self.graph, self.k)
        x = self.layers(x)
        if not remove_last_layer:
            x = self.fc(x)
        return x
    
