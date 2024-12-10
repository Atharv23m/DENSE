import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import random

def load_data(dataset, data_dir):
    if dataset == "custom":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]),
        }

        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    else:
        raise NotImplementedError

    X_train, y_train = [s[0] for s in train_dataset.samples], train_dataset.targets
    X_test, y_test = [s[0] for s in test_dataset.samples], test_dataset.targets

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, partition, beta=0.4, num_users=5, data_dir=None):
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset, data_dir)
    data_size = len(y_train)
    indices = list(range(data_size))
    random.shuffle(indices)
    shard_size = data_size // num_users
    net_dataidx_map = {i: indices[i * shard_size: (i + 1) * shard_size] for i in range(num_users)}

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, traindata_cls_counts