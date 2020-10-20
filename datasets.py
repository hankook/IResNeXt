import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch
from torch.utils.data import DataLoader

class SampledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        assert len(dataset) >= len(indices)

    def __getitem__(self, index):
        return self.dataset.__getitem__(self.indices[index])

    def __len__(self):
        return len(self.indices)

def get_val_indices(datadir, size, force=False):
    filename = os.path.join(datadir, 'val_indices_%d.txt'%size)
    if os.path.isfile(filename) and not force:
        with open(filename, 'r') as f:
            indices = [int(i.strip()) for i in f.readlines()]
        assert len(indices) == size
    else:
        indices = sorted(random.sample(list(range(50000)), size))
        with open(filename, 'w') as f:
            f.write('\n'.join([str(i) for i in indices]))
    return indices

def get_dataset(name, **kwargs):
    datadir = os.path.expanduser('~/data/' + name)
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    if name == 'cifar10' or name == 'cifar100':
        mean = np.array([125.3, 123.0, 113.9]) / 255.0
        std = np.array([63.0,  62.1,  66.7]) / 255.0

        if name == 'cifar10':
            data = torchvision.datasets.CIFAR10
        else:
            data = torchvision.datasets.CIFAR100

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        val_size = kwargs.get('val_size', 5000)
        val_indices = get_val_indices(datadir, val_size)
        train_indices = sorted(list(set(range(50000))-set(val_indices)))

        trainset = data(datadir, train=True, transform=train_transform, download=True)
        trainset = SampledDataset(trainset, train_indices)

        valset = data(datadir, train=True, transform=test_transform, download=True)
        valset = SampledDataset(valset, val_indices)

        testset = data(datadir, train=False, transform=test_transform, download=True)

        return trainset, valset, testset
    else:
        raise Exception('unknown dataset: %s'%dataset)

def get_dataloader(datasets, batch_size, shuffle=True):
    return [DataLoader(d, batch_size=batch_size, shuffle=True, num_workers=4)
                for d in datasets]
