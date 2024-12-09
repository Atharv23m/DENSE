import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import random

def load_data(dataset):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
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

    X_train, y_train = train_dataset.samples, train_dataset.targets
    X_test, y_test = test_dataset.samples, test_dataset.targets

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, partition, beta=0.4, num_users=5):
    X_train, y_train, X_test, y_test, train_dataset, test_dataset = load_data(dataset)
    data_size = len(y_train)

    if partition == "custom":
        net_dataidx_map = {}
        for i in range(num_users):
            client_dir = os.path.join('/path/to/your/dataset', f'Client {i+1}')
            client_dataset = datasets.ImageFolder(client_dir, transform=train_dataset.transform)
            net_dataidx_map[i] = [train_dataset.samples.index(sample) for sample in client_dataset.samples]

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return train_dataset, test_dataset, net_dataidx_map, train_data_cls_counts