import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
from tqdm import tqdm
def train():
    device = 'cuda:0'

    data_transform = {
        "train":transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val":transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_data = torchvision.datasets.CIFAR10(
        root="../data/mnist/",  # 训练数据保存路径
        train=True,
        # transform=data_transform["train"], 
        transform=torchvision.transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
        download=False,  # 是否需要下载
    )
    test_data = torchvision.datasets.MNIST(root="../../net_test/mnist/",
                                         transform=data_transform["val"], 
                                          train=False)
    print(test_data.test_data.size())
    print(train_data.train_data.size())

    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
         batch_size=batch_size, shuffle=True, num_workers=4)
    train_bar = tqdm(train_loader, file=sys.stdout)
    for i in enumerate(train_bar):
        print(i)
        break
if  __name__ == '__main__':
    train()