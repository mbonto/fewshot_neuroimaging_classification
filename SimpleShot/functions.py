import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import random
from numpy import linalg as LA
from scipy.stats import mode
import collections
import tqdm

from architecture import *


def batch_shape(model_name, x):
    """
    Return a batch x with a relevant shape for a given model. For example,
    # MLP
    x = torch.ones(5, 360) # batch, features
    # Conv1d
    x = torch.ones(5, 1, 360) # batch, channel, features
    # GNN
    x = torch.ones(5, 360) # batch, features
    graph = torch.ones(360, 360)
    # Resnet3d
    x = torch.ones(5, 1, 100, 100, 100) # batch, channel, dim1, dim2, dim3
    """    
    if model_name == 'MLP' or model_name == 'GNN':
        return torch.squeeze(x, dim=1)
    elif model_name == 'Resnet3d':
        return torch.unsqueeze(x, dim=1)
    else:
        return x
 
    
def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    torch.save(state, folder + filename)
    if is_best:
        shutil.copyfile(folder + filename, folder + '/model_best.pth.tar')

        
def load_checkpoint(model, save_path, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(save_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])
    
    
def compute_accuracy(outputs, y):       
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y).sum().item()
    return correct


# Train a model for 1 epoch and return its loss.
def train(model, criterion, optimizer, data_loader, use_cuda):
    """ Train a neural network for one epoch.
    """
    model.train()
    epoch_loss = 0.
    epoch_acc = 0.
    epoch_total = 0.
    for i, (x, y) in enumerate(data_loader):
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # Adapt the batch shape to the model.
        x = batch_shape(model.name, x)
        # Zero the parameter gradients.
        optimizer.zero_grad()
        # Forward + backward + optimize.
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()  
        optimizer.step()
        acc = compute_accuracy(outputs.clone().detach(), y)
        # Statistics.
        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_total += y.size(0)
    return epoch_loss / (i+1), epoch_acc / epoch_total


# Evaluate a model on the validation / test set.
def episodic_evaluation(model, data_loader, sampler_infos, use_cuda):
    """
    Return the average accuracy on few-shot tasks (called episodes).
    A task contains training samples with known labels and query
    samples. The accuracy is the number of times we correctly
    predict the labels of the query samples.
    
    To attribute a label to a new sample, we consider the outputs of the
    penultimate layer of model. A label is represented by the average
    outputs of its training samples. A new sample is labeled
    in function of the closest label-representative.
    """
    model.eval()
    n_way = sampler_infos[1]
    n_shot = sampler_infos[2]
    epoch_acc = 0.
    total = 0.
    
    with torch.no_grad():
        # Iterate over several episodes.
        for i, (x, y) in enumerate(tqdm.tqdm(data_loader)):
            # print(i, end='\r')
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            # Adapt the batch shape to the model.
            x = batch_shape(model.name, x)
            # Retrieve the outputs of the penultimate layer of model of all
            # samples.         
            outputs = model(x, remove_last_layer=True)
            # print('outputs shape', outputs.shape)       
            training = outputs[:n_way*n_shot]
            query = outputs[n_way*n_shot:]
            train_labels = y[:n_way*n_shot]
            query_labels = y[n_way*n_shot:]

            # Compute the vector representative of each class.
            training = training.reshape(n_way, n_shot, -1).mean(1)
            train_labels = train_labels[::n_shot]

            # Find the labels of the query samples.
            scores = cosine_score(training, query)
            pred_labels = torch.argmin(scores, dim=1)
            pred_labels = torch.take(train_labels, pred_labels)
            
            # Compute the accuracy.
            acc = (query_labels == pred_labels).float().sum()
            epoch_acc += acc
            total += query_labels.size(0)
            del training, query
    return epoch_acc / total


# Compute similarities between two sets of vectors.
def cosine_score(X, Y):
    """
    Return a score between 0 and 1 (0 for very similar, 1 for not similar at all)
    between all vectors in X and all vectors in Y. As the score is based on the
    cosine similarity, all vectors are expected to have positive values only.
    
    Parameters:
        X -- set of vectors (number of vectors, vector size).
        Y -- set of vectors (number of vectors, vector size).
    """
    scores = 1. - F.cosine_similarity(Y[:, None, :], X[None, :, :], dim=2)
    return scores


def sample_case(data, n_shot, n_way, n_query):
    """
    Return the training and test data of a few-shot task.
    
    Parameters:
        data -- dict whose keys are labels and values are the samples associated with.
        n_way -- int, number of classes
        n_shot -- int, number of training examples per class.
        n_query -- int, number of test examples per class.
    """
    # Randomly sample n_way classes.
    classes = random.sample(list(data.keys()), n_way)
    train_data = []
    test_data = []
    test_labels = []
    train_labels = []
    # For each class, randomly select training and test examples.
    for label in classes:
        samples = random.sample(data[label], n_shot + n_query)
        train_labels += [label] * n_shot
        test_labels += [label] * n_query
        train_data += samples[:n_shot]
        test_data += samples[n_shot:]
    train_data = np.array(train_data).astype(np.float32)
    test_data = np.array(test_data).astype(np.float32)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    return train_data, test_data, train_labels, test_labels


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval.
    
    Parameters:
        data -- Array containing an estimation of a value obtained across different data samples.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def metric_class_type(train_data, test_data, train_labels, test_labels, n_way, n_shot, base_mean=None, norm_type='CL2N'):
    """
    Centering: The mean vector of the base examples is substracted from the (train_data + test_data) examples.
    L2-normalization: Each (train_data + test_data) example is divided by its L2-norm.
    
    Parameters:
        train_data -- training examples of the few-shot task, of size (number of trainig examples, number of parameters).
                   Examples associated with the same class follow each other.
        test_data -- test examples of the few-shot task, of size (number of test examples, number of parameters).
                 Examples associated with the same class follow each other.
        train_labels -- labels of the training examples.
        test_labels -- labels of the test examples.
        n_way -- number of classes.
        n_shot -- number of training examples per class.
        base_mean -- mean vector of the base examples.
        norm_type -- None, 'L2N' or 'CL2N'. 'L2N' stands for L2-normalization. 'CL2N' for centering and L2-normalization.
    """
    # Normalize the data samples.
    if norm_type == 'CL2N':
        train_data = train_data - base_mean
        train_data = train_data / LA.norm(train_data, 2, 1)[:, None]
        test_data = test_data - base_mean
        test_data = test_data / LA.norm(test_data, 2, 1)[:, None]
    elif norm_type == 'L2N':
        train_data = train_data / LA.norm(train_data, 2, 1)[:, None]
        test_data = test_data / LA.norm(test_data, 2, 1)[:, None]
    # Compute the vectors representative of each class.
    train_data = train_data.reshape(n_way, n_shot, -1).mean(1)
    train_labels = train_labels[::n_shot]
    
    # Find the labels of the test samples.
    # Look at the Euclidean distances between class representatives and test examples.
    subtract = train_data[:, None, :] - test_data
    # print(subtract.shape, 'n_way, n_test*n_way, n_param')
    distance = LA.norm(subtract, 2, axis=-1)
    # print(distance.shape, 'n_way, n_test*n_way')
    # Each test example is labeled as its closest representative. Idx is of size (1, n_exs).
    num_NN = 1
    pred_labels = np.argpartition(distance, num_NN, axis=0)[:num_NN]
    pred_labels = np.take(train_labels, pred_labels)
    
    # Output the value which is most present in pred_labels (which is not necessary here).
    out = mode(pred_labels, axis=0)[0]
    out = out.astype(int)
    
    # Compute the accuracy.
    acc = (out == test_labels).mean()
    return acc


def extract_feature(train_loader, val_loader, model, tag='last'):
    """
    Return out_mean, which is the average of the examples in train_loader used to train model.
    Return output_dict, which is a dictionary whose keys are the labels of the classes in val_loader,
    and values are the outputs of the penultimate layer of model computed on the examples in val_loader.
    """
    model.eval()
    with torch.no_grad():
        # Get the average of the examples used to train the model.
        out_mean = []
        for x, _ in train_loader:
            # Adapt the batch shape to the model.
            x = batch_shape(model.name, x)
            outputs = model(x, remove_last_layer=True)  ##
            out_mean.append(outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)

        output_dict = collections.defaultdict(list)
        for x, y in val_loader:
            # Adapt the batch shape to the model.
            x = batch_shape(model.name, x)
            # Compute the output of the penultimate layer of model.
            outputs = model(x, remove_last_layer=True)
            outputs = outputs.cpu().data.numpy()
            # Save it.
            for out, label in zip(outputs, y):
                output_dict[label.item()].append(out)
        return out_mean, output_dict


def meta_evaluate(data, base_mean, n_episode=10000, n_way=5, n_shot=1, n_query=15):
    """
    Return the average accuracy and 95% confidence interval on few-shot tasks.
    
    Parameters:
        data -- dict whose keys are labels and values are the samples associated with.
        base_mean -- mean vector of the base examples.
        n_episode -- number of tasks the results are averaged on.
        n_way -- number of classes in a few-shot task.
        n_shot --  number of training examples in a few-shot task.
        n_query -- number of test examples in a few-shot task.
    """
    un_list = []
    l2n_list = []
    cl2n_list = []
    for _ in range(n_episode):
        train_data, test_data, train_labels, test_labels = sample_case(data, n_shot, n_way, n_query)
        
        acc = metric_class_type(train_data, test_data, train_labels, test_labels, n_way, n_shot, base_mean=base_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, train_labels, test_labels, n_way, n_shot, base_mean=base_mean,
                                norm_type='L2N')
        l2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, train_labels, test_labels, n_way, n_shot, base_mean=base_mean,
                                norm_type='UN')
        un_list.append(acc)
    
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf


def do_extract_and_evaluate(model, train_loader, val_loader, save_path):
    model = model.cpu()
    load_checkpoint(model, save_path, 'last')
    out_mean, out_dict = extract_feature(train_loader, val_loader, model, 'last')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, n_shot=1, n_episode=10000)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, n_shot=5, n_episode=10000)
    print(
        'Meta Test: LAST\nfeature\t\tUN\t\tL2N\t\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

    load_checkpoint(model, save_path, 'best')
    out_mean, out_dict = extract_feature(train_loader, val_loader, model, 'best')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, n_shot=1, n_episode=10000)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, n_shot=5, n_episode=10000)
    print(
        'Meta Test: BEST\nfeature\t\tUN\t\tL2N\t\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))

    
def get_model_parameters(model_name, num_layers=None, num_feat=None, num_classes=None, n_step=None):
    if model_name == 'MLP':
        model_parameters = [num_layers, num_feat, num_classes]
    elif model_name == 'Conv1d':
        model_parameters = [num_layers, num_feat, num_classes, 1]
    elif model_name == 'GNN':
        model_parameters = [num_classes, n_step, num_layers, num_feat]
    return model_parameters


def get_model(model_name, model_parameters):
    if model_name == 'MLP':
        model = MLP(*model_parameters)
    elif model_name == 'Conv1d':
        model = Conv1d(*model_parameters)
    elif model_name == 'GNN':
        model = GNN(*model_parameters)
    return model


def get_variables(model_name):
    if model_name in ['MLP', 'GNN', 'Conv1d']:
        parcel = True
        batch_size = 128
    return parcel, batch_size


def get_optimizer(model, lr=0.1, n_epoch=1, lr_gamma=0.1, weight_decay=1e-4):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * n_epoch), int(0.9 * n_epoch)], gamma=lr_gamma)
    return optimizer, scheduler


def do_extract_and_evaluate_simplified(model, train_loader, val_loader, save_path):
    model = model.cpu()
    if model.name == 'GNN':
        model.graph = model.graph.cpu()
    load_checkpoint(model, save_path, 'best')
    out_mean, out_dict = extract_feature(train_loader, val_loader, model, 'best')
    accuracy_info_shot1 = meta_evaluate_simplified(out_dict, out_mean, n_shot=1, n_episode=10000)
    accuracy_info_shot5 = meta_evaluate_simplified(out_dict, out_mean, n_shot=5, n_episode=10000)
    
    return accuracy_info_shot1[0], accuracy_info_shot1[1], accuracy_info_shot5[0], accuracy_info_shot5[1]


def meta_evaluate_simplified(data, base_mean, n_episode=10000, n_way=5, n_shot=1, n_query=15):
    """
    Return the average accuracy and 95% confidence interval on few-shot tasks.
    
    Parameters:
        data -- dict whose keys are labels and values are the samples associated with.
        base_mean -- mean vector of the base examples.
        n_episode -- number of tasks the results are averaged on.
        n_way -- number of classes in a few-shot task.
        n_shot --  number of training examples in a few-shot task.
        n_query -- number of test examples in a few-shot task.
    """
    cl2n_list = []

    for _ in range(n_episode):
        train_data, test_data, train_labels, test_labels = sample_case(data, n_shot, n_way, n_query)
        
        acc = metric_class_type(train_data, test_data, train_labels, test_labels, n_way, n_shot, base_mean=base_mean,
                                norm_type='CL2N')
        cl2n_list.append(acc)
    
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return cl2n_mean, cl2n_conf