import torch.nn as nn
import torch




class BigBlock(nn.Module):

    '''
    resnet网络层数50,101,152的残差模块,同一残差模块内保持shape不变,最后一次卷积shape变为上一次的4倍。
    
    '''
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BigBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=1,
                                stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)


        # 卷积核属性：3*3*channels   padding = 1 以保持图片shape不变
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channel=out_channel, kernel_size=3,    
                                stride=stride, padding=1) 
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 卷积核属性：1*1*(channels*4)  padding = 0 以保持图片shape不变 维度变为原来的4倍
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channel=out_channel*self.expansion, kernel_size=1,    
                                stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)  #是否进行覆盖运算

    def forward(self,x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.relu(x+identity)

        return x



class ResNet(nn.Module):


    '''
    ResNet 整体流程

    block: 使用的是哪个残差模块。
    blocks_num: 每个网络层使用次数，以列表形式掺入。
    
    '''

    def __init__(self,
                block: BigBlock,
                blocks_num: list
                ):
        super(ResNet,self).__init__()
        self.in_channel = 64

        '''
        第一此卷积,使用7*7卷积核,步长为2,为了使shape缩小为原来的一半,padding为3,输入通道为RGB3通道,
        输出通道是64
        '''
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,
                             stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        #池化，缩小一半shape
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self.make_layer(block,64,blocks_num[0])


    def make_layer(self,block,channel,times:int):
        print(1)
        pass