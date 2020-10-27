# Inspired from code from https://github.com/mileyan/simple_shot. It has been modified.
import os
import numpy as np
from nilearn.image import load_img
import torch
import torchvision.transforms as transforms
from torch.utils.data import Sampler

__all__ = ['get_dataloader', 'DatasetFolder']


class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, transform=None, parcel=True):
        assert split_type in ['train', 'test', 'val']
        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.label_map = label_map
        self.length = len(self.data)
        self.parcel = parcel

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert os.path.isfile(self.root + self.data[index] + '.npz')
        
        if self.parcel:
            npzfile = self.root + '/' + self.data[index] + '.npz'
            img = np.load(npzfile)['X']
        else:
            niifile = self.root + '/' + self.data[index] + '.nii.gz'
            img = load_img(niifile).get_data()
                    
        label = self.labels[index]
        label = int(label)
        
        img = torch.from_numpy(img).type(torch.float)
        if self.transform:
            img = self.transform(img)
        
        return img, label


normalization = transforms.Normalize(mean=[0.0],
                                std=[np.sqrt(2)])

def normalize(parcel):
    if parcel:
        return None
    else:
        return normalization


def get_dataloader(split, data, parcel, split_dir, meta=False, batch_size=None, sampler=None, sampler_infos=None):
    """
    Return a dataloader.
    
    Parameters:
        split -- 'train, 'val' or 'test'.
        data -- path towards the data.
        parcel -- boolean, use parcellated images or whole-voxel images.
        split_dir -- path towards the split files.
        meta -- boolean. If True, a batch corresponds to a task with a given number of classes (n_way),
                training examples per class (n_shot), test examples per class (n_query).
                If False, a batch corresponds to a given number of images chosen among the whole dataset.
        batch_size -- Number of elements per batch. Only used if meta == False.
        sampler -- Sampler.WeightedRandomSampler, allows to balance an epoch when a dataset is imbalanced.
                   Only used if meta == False.
        sampler_infos -- list containing 4 values [number of batch per epoch, n_way, n_shot, n_query].
                         Only used if meta == True.
        
    """
    # We normalize the data such that they have 0 mean and 1 std.
    transform = normalize(parcel)
    
    sets = DatasetFolder(data, split_dir, split, transform, parcel)
    if meta:
        sampler = CategoriesSampler(sets.labels, *sampler_infos)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler, num_workers=3, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=sampler)
    return loader


# Define a batch sampler to load a dataset where each batch corresponds to
# a task (which contains 'n_way' classes, 'n_shot' examples per class whose label is given,'n_query' examples per class for whose label is to guess).
class CategoriesSampler(Sampler):

    def __init__(self, label, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        self.m_ind = []
        unique = np.unique(label)
        unique = np.sort(unique)
        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            batch_base = []
            batch_query = []
            # Randomly select n_way classes.
            classes = torch.randperm(len(self.m_ind))[:self.n_way]
            # For each class, randomly select elements for training
            # (base) and for test (query).
            for c in classes:
                l = self.m_ind[c.item()]
                pos = torch.randperm(l.size()[0])
                batch_base.append(l[pos[:self.n_shot]])
                batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
            batch = torch.cat(batch_base + batch_query)
            yield batch
            
