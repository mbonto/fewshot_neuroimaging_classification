# Inspired from code from https://github.com/yhu01/PT-MAP. It has been modified.
import collections
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA
import tqdm

from functions import load_checkpoint, batch_shape, compute_confidence_interval


# Loading data.
def centerDatas(datas, n_lsamples):
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] - datas[:, :n_lsamples].mean(1, keepdim=True)
    datas[:, :n_lsamples] = datas[:, :n_lsamples, :] / torch.norm(datas[:, :n_lsamples, :], 2, 2)[:, :, None]
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] - datas[:, n_lsamples:].mean(1, keepdim=True)
    datas[:, n_lsamples:] = datas[:, n_lsamples:, :] / torch.norm(datas[:, n_lsamples:, :], 2, 2)[:, :, None]
   
    return datas

def scaleEachUnitaryDatas(datas):
  
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms


def QRreduction(datas):
    
    ndatas = torch.qr(datas.permute(0,2,1)).R
    ndatas = ndatas.permute(0,2,1)
    return ndatas


class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
# Gaussian Model.
class GaussianModel(Model):
    def __init__(self, n_ways, lam, n_nfeat, n_shot, n_queries, n_runs):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None  # centroids
        self.lam = lam
        self.n_nfeat = n_nfeat
        self.n_shot = n_shot
        self.n_ways = n_ways
        self.n_queries = n_queries
        self.n_runs = n_runs
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self, ndatas, labels):
        # print(ndatas.shape, '1, 100, 100')
        self.mus = ndatas.reshape(self.n_runs, self.n_shot + self.n_queries, self.n_ways, self.n_nfeat)[:,:self.n_shot,].mean(1)
        self.l = labels.float().reshape(self.n_runs, self.n_shot + self.n_queries, self.n_ways, 1)[:,:self.n_shot,].mean(1)
        # print(self.l, '0, 1, 2, 3, 4')
        # print(self.mus.shape, '1, 5, 100')

    def updateFromEstimate(self, estimate, alpha):          
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):      
        r = r.cpu()
        c = c.cpu()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)                                     
        u = torch.zeros(n_runs, n).cpu()
        maxiters = 1000
        iters = 1
        # Normalize the matrix.
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, ndatas, labels):
        n_lsamples = self.n_ways * self.n_shot
        n_usamples = self.n_ways * self.n_queries
        n_samples = n_lsamples + n_usamples
        
        # Compute the squared distance to centroids.
        ndatas = ndatas.cpu()
        self.mus = self.mus.cpu()
        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        p_xj = torch.zeros_like(dist)
        r = torch.ones(self.n_runs, n_usamples)
        c = torch.ones(self.n_runs, self.n_ways) * self.n_queries
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-6)
        p_xj[:, n_lsamples:] = p_xj_test
        
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2, labels[:, :n_lsamples].unsqueeze(2), 1)
        
        return p_xj

    def estimateFromMask(self, mask, ndatas):
        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

          
# MAP.

class MAP:
    def __init__(self, alpha=None):        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas, labels, n_lsamples, n_runs):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() * 1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, ndatas, labels, n_lsamples, n_runs, epochInfo=None):
        p_xj = model.getProbas(ndatas, labels)
        self.probas = p_xj
        
        if self.verbose:
            pass
            # print("accuracy from filtered probas", self.getAccuracy(self.probas))
        
        m_estimates = model.estimateFromMask(self.probas, ndatas)
               
        # Update centroids.
        model.updateFromEstimate(m_estimates, self.alpha)

        if self.verbose:
            op_xj = model.getProbas(ndatas, labels)
            acc = self.getAccuracy(op_xj, labels, n_lsamples, n_runs)
            # print("output model accuracy", acc)
        
    def loop(self, model, ndatas, labels, n_lsamples, n_runs, n_epochs=20):
        
        self.probas = model.getProbas(ndatas, labels)
        if self.verbose:
            pass
            # print("initialisation model accuracy", self.getAccuracy(self.probas))

#         if self.progressBar:
#             if type(self.progressBar) == bool:
#                 pb = tqdm.tqdm(total = n_epochs)
#             else:
#                 pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}  lr_m: {:0.3f}".format(epoch, self.alpha))
            self.performEpoch(model, ndatas, labels, n_lsamples, n_runs, epochInfo=(epoch, n_epochs))
            #if (self.progressBar): pb.update()
        
        # Get final accuracy and return it.
        op_xj = model.getProbas(ndatas, labels)
        acc = self.getAccuracy(op_xj, labels, n_lsamples, n_runs)
        return acc
    
    
def PT_plus_MAP(ndatas, labels, n_ways, n_shot, n_queries, n_runs=1):
    # Power transform.
    beta = 0.5
    ndatas[:,] = torch.pow(ndatas[:,]+1e-6, beta)
    # Reduce the number of features with a the QR reduction.
    ndatas = QRreduction(ndatas)
    # Normalization.
    n_nfeat = ndatas.size(2)
    ndatas = scaleEachUnitaryDatas(ndatas)
    ndatas = centerDatas(ndatas, n_ways * n_shot)
    # Switch to cuda.
    ndatas = ndatas.cpu()
    labels = labels.cpu()
    # MAP.
    lam = 10
    model = GaussianModel(n_ways, lam, n_nfeat, n_shot, n_queries, n_runs)
    model.initFromLabelledDatas(ndatas, labels)

    alpha = 0.2
    optim = MAP(alpha)

    optim.verbose=False
    optim.progressBar=True

    acc_test = optim.loop(model, ndatas, labels, n_ways * n_shot, n_runs, n_epochs=20)
    # print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100*x for x in acc_test)))
    
    return acc_test


def sample_case_PT_plus_MAP(data, n_shot, n_way, n_query):
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
    i = 0
    for label in classes:
        samples = random.sample(data[label], n_shot + n_query)
        train_labels += [i] * n_shot
        test_labels += [i] * n_query
        train_data += samples[:n_shot]
        test_data += samples[n_shot:]
        i += 1

    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels)
    
    train_data2 = []
    train_labels2 = []

    for d in range(n_shot):
        for i in range(n_way):
            train_data2.append(train_data[d+i*n_shot])
            train_labels2.append(train_labels[d+i*n_shot])
    
    train_data2 = np.array(train_data2).astype(np.float32)
    train_labels2 = np.array(train_labels2)
    
    return train_data2, test_data, train_labels2, test_labels


def extract_feature_PT_plus_MAP(val_loader, model, tag='last'):
    """
    Return out_mean, which is the average of the examples in train_loader used to train model.
    Return output_dict, which is a dictionary whose keys are the labels of the classes in val_loader,
    and values are the outputs of the penultimate layer of model computed on the examples in val_loader.
    """
#     save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
#     if os.path.isfile(save_dir + '/output.plk'):
#         data = load_pickle(save_dir + '/output.plk')
#         return data
#     else:
#         if not os.path.isdir(save_dir):
#             os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        output_dict = collections.defaultdict(list)
        for x, y in val_loader:
            # Adapt the batch shape to the model.
            x = batch_shape(model.name, x)
            # Compute the output of the penultimate layer of model.
            outputs = model(x, remove_last_layer=True)  ##
            outputs = outputs.cpu().data.numpy()
            # Save it.
            for out, label in zip(outputs, y):
                output_dict[label.item()].append(out)
#         save_pickle(save_dir + '/output.plk', all_info)
        return output_dict

def do_extract_and_evaluate_simplified_PT_plus_MAP(model, test_loader, save_path):
    model = model.cpu()
    if model.name == 'GNN':
        model.graph = model.graph.cpu()
    load_checkpoint(model, save_path, 'best')
    out_dict = extract_feature_PT_plus_MAP(test_loader, model, 'best')
    accuracy_info_shot1 = loop_on_episodes(out_dict, n_shot=1, n_episode=10000)
    accuracy_info_shot5 = loop_on_episodes(out_dict, n_shot=5, n_episode=10000)
    
    return  accuracy_info_shot1[0], accuracy_info_shot1[1], accuracy_info_shot5[0], accuracy_info_shot5[1]

# Generate the test tasks.
def loop_on_episodes(dataset, n_episode=10000, n_way=5, n_shot=1, n_query=15):
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
        # Sample a task.
        train_data, test_data, train_labels, test_labels = sample_case_PT_plus_MAP(dataset, n_shot, n_way, n_query)
        
        # Reorganize the data.
        train_data = torch.unsqueeze(torch.from_numpy(train_data), dim=0).cpu()
        train_labels = torch.unsqueeze(torch.from_numpy(train_labels), dim=0).cpu()

        test_data = torch.unsqueeze(torch.from_numpy(test_data), dim=0).cpu()
        test_labels = torch.unsqueeze(torch.from_numpy(test_labels), dim=0).cpu()
        data = torch.cat((train_data, test_data), dim=1)
        labels = torch.cat((train_labels, test_labels), dim=1)
        
        # Launch PT_plus_MAP.
        mean, conf = PT_plus_MAP(data, labels, n_way, n_shot, n_query)
        
        cl2n_list.append(mean)
        
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return cl2n_mean, cl2n_conf