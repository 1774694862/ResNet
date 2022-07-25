from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
import torchvision

class DataInterface():

    def __init__(self,**args: argparse.Namespace) -> None:
        self.args = args
        self.batch_size = self.args['batch_size']
        self.num_workers = self.args['num_workers']


        self.data_transform = {
            "train":transforms.Compose([transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "val":transforms.Compose([transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    def train_dataloader(self) -> DataLoader:
        train_data = torchvision.datasets.CIFAR10(
            root="../data/",  # 训练数据保存路径
            train=True,
            transform=self.data_transform["train"], 
            # transform=torchvision.transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
            download=False  # 是否需要下载
        )

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )    
        return train_loader
    def test_dataloader(self) -> DataLoader:
        test_data = torchvision.datasets.CIFAR10(
            root="../data/",
            transform=self.data_transform["val"], 
            train=False
        )
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )  
        return  test_loader