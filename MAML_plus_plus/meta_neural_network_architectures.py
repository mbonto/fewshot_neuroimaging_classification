import numbers
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy import io as sio



def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item
    return output_dict



class MetaConv1dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv1dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv1d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out
    
    def restore_backup_stats(self):
        pass



class Conv1d(nn.Module):
    def __init__(self, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer MLP network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        
        Parameters:
            num_output_classes -- The number of output classes of the network.
            args -- A named tuple containing the system's hyperparameters.
            device -- The device to run this on.
            meta_classifier -- A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(Conv1d, self).__init__()
        self.device = device
        self.args = args
        self.num_features = args.num_features
        self.num_layers = args.num_stages  ## number of layers
        self.num_output_classes = num_output_classes

        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['conv0'] = MetaConv1dLayer(in_channels=1, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, use_bias=True, groups=1, dilation_rate=1)
        for i in range(1, self.num_layers):
            self.layer_dict['conv{}'.format(i)] = MetaConv1dLayer(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, use_bias=True, groups=1, dilation_rate=1)

        self.layer_dict['linear'] = MetaLinearLayer(360, self.num_output_classes, use_bias=True, activation=False)
        

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        # print(x.shape) # 5, 1, 360, 1
        out = torch.squeeze(x, dim=-1)
        # print('input shape is: ' + str(out.shape))

        for i in range(self.num_layers):
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)])
            out = F.relu(out)
        
        # Average pooling by "pixels".
        out = torch.mean(out, dim=1)
        out = torch.squeeze(out, dim=-2)
        # Or keep all coefficients until the end.
        # out = out.view(out.size(0), -1)
        
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_layers):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()



class MetaLinearLayer(nn.Module):
    def __init__(self, num_in_filters, num_out_filters, use_bias, activation):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_out_filters, num_in_filters))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_out_filters))
        self.activation = activation

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        
        out = F.linear(input=x, weight=weight, bias=bias)
        if self.activation:
           out = F.relu(out)
        return out
    
    def restore_backup_stats(self):
        pass



### New architectures ###
class MLP(nn.Module):
    def __init__(self, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer MLP network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        
        Parameters:
            num_output_classes -- The number of output classes of the network.
            args -- A named tuple containing the system's hyperparameters.
            device -- The device to run this on.
            meta_classifier -- A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(MLP, self).__init__()
        self.device = device
        self.args = args
        self.num_features = args.num_features
        self.num_layers = args.num_stages  ## number of hidden layers
        self.num_output_classes = num_output_classes
        self.meta_classifier = meta_classifier
        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['linear0'] = MetaLinearLayer(360, self.num_features, use_bias=True, activation=True)
        for i in range(1, self.num_layers):
            self.layer_dict['linear{}'.format(i)] = MetaLinearLayer(self.num_features, self.num_features, use_bias=True, activation=True)
        self.layer_dict['linear'] = MetaLinearLayer(self.num_features, self.num_output_classes, use_bias=True, activation=False)
        

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        # print(x.shape) 5, 1, 360, 1
        out = torch.squeeze(x, dim=-1)
        out = torch.squeeze(out, dim=1)
        # print('input shape is: ' + str(out.shape))

        for i in range(self.num_layers):
            out = self.layer_dict['linear{}'.format(i)](out, params=param_dict['linear{}'.format(i)])
        out = self.layer_dict['linear'](out, param_dict['linear'])

        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_layers):
            self.layer_dict['linear{}'.format(i)].restore_backup_stats()



def compute_normalized_adjacency(adj_matrix, threshold=True):
    norm_adj = torch.from_numpy(adj_matrix)
    if threshold:
        # We keep the 1% highest connections (threshold value of 0.945 for the considered brain graph).
        mask = (norm_adj > 0.945).type(torch.float)
        norm_adj = norm_adj * mask
    # Add the identity matrix to the adjacency matrix.
    norm_adj = torch.add(norm_adj, torch.eye(norm_adj.shape[0]))
    # Compute the degree matrix by summing the columns.
    degree = torch.sum(norm_adj, dim=1)
    degree = torch.pow(degree, -1/2)
    degree = torch.diag(degree)
    # Compute a normalized adjacency matrix.
    norm_adj = torch.matmul(torch.matmul(degree, norm_adj), degree)
    return norm_adj.type(torch.float)


def propagation(features, graph, k=1):
    """
    Return the features propagated k times on the graph.
    
    Params:
        features -- tensor of size (batch, num_features).
        graph -- adjacency matrix of size (num_features, num_features).
        k -- number of times the features are propagated, integer.
    """
    for _ in range(k):
        features = torch.matmul(features, graph)
    return features


class GNN(nn.Module):
    def __init__(self, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer GNN network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        
        Parameters:
            num_output_classes -- The number of output classes of the network.
            args -- A named tuple containing the system's hyperparameters.
            device -- The device to run this on.
            meta_classifier -- A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(GNN, self).__init__()
        self.device = device
        self.args = args
        self.num_features = args.num_features
        self.num_layers = args.num_stages
        self.num_output_classes = num_output_classes

        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)
     
        # Load the structural connectivity matrix.
        # The connections are computed between the 360 regions defined in glasser. They are established from 
        # structural MRI (T1, DWI). Self-connections are set to zero.
        # DWI: Tractography looking at the strength of the white fibers between regions, averaged on 56 subjects. 
        connectivity_matrix_path = args.connectivity_matrix_path
        connectivity = sio.loadmat(connectivity_matrix_path)['SC_avg56']      
        self.graph = compute_normalized_adjacency(connectivity).cuda()
        
        # Number of diffusion step.
        self.k = 1
    
    
    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['linear0'] = MetaLinearLayer(360, self.num_features, use_bias=True, activation=True)
        for i in range(1, self.num_layers):
            self.layer_dict['linear{}'.format(i)] = MetaLinearLayer(self.num_features, self.num_features, use_bias=True, activation=True)
        self.layer_dict['linear'] = MetaLinearLayer(self.num_features, self.num_output_classes, use_bias=True, activation=False)
        

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        if params is not None:
            params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        # Reshape the data.
        # print(x.shape) 5, 1, 360, 1
        out = torch.squeeze(x, dim=-1)
        out = torch.squeeze(out, dim=1)
        # print('input shape is: ' + str(out.shape))
        
        # Propagate the signal on the graph.
        out = propagation(out, self.graph, self.k)
        
        # Go through the MLP network.
        for i in range(self.num_layers):
            out = self.layer_dict['linear{}'.format(i)](out, params=param_dict['linear{}'.format(i)])

        out = self.layer_dict['linear'](out, param_dict['linear'])
        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_layers):
            self.layer_dict['linear{}'.format(i)].restore_backup_stats()
 
