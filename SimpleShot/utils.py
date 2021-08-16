import numpy as np
import torch
import torch.utils.data.sampler as spler
import configargparse

from data import DatasetFolder


def create_sampler(IBC_path, split_dir, split):
    dataset = DatasetFolder(IBC_path, split_dir, split, parcel=True)
    class_sample_count = np.array(
        [len(np.where(np.array(dataset.labels) == t)[0]) for t in np.unique(dataset.labels)])

    minimum = min(class_sample_count)
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in dataset.labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = spler.WeightedRandomSampler(samples_weight, int(minimum*len(np.unique(dataset.labels))), replacement=False)
    return sampler


def parser_args():
    parser = configargparse.ArgParser(description='Transfer-based few-shot on fMRI data.')
    ### Model
    parser.add_argument('--model-name', type=str, required=True,
                    help='name of the architecture used as a backbone')
    parser.add_argument('--num_shot', type=int, required=True,
                    help='number of examples per class in a few-shot task')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='number of layers in the network')
    parser.add_argument('--num-feat', type=int, default=512,
                        help='number of features in the network')
    parser.add_argument('--num-classes', type=int, default=61,
                        help='use all data to train the network')
    parser.add_argument('--subsample', type=int, default=1,
                        help='subsampling factor of the input of Resnet3D')
    parser.add_argument('--kernel', type=int, default=1,
                        help='kernel size of the convolutions of Conv1d')
    parser.add_argument('--n-step', type=int, default=1,
                        help='number of times the input is propagated in GNN')
    parser.add_argument('--inplanes', type=int, default=64,
                        help='number of initial feature maps in Resnet3d')
    ### Optimization
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--n-epoch', default=90, type=int,
                        help='number of epochs')
    return parser.parse_args()
