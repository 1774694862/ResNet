import torch.nn as nn
import torch

class BigBlock(nn.Module):
    '''
    resnet网络层数50,101,152的残差模块,同一残差模块内,最后一次卷积channel变为上一次的4倍。
    downsample:同等映射下,传进来的shape不一致时需要改变shape。
    '''
    expansion = 4
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BigBlock, self).__init__()

        #卷积之后，如果要接BN操作，最好是不设置偏置，因为不起作用，而且占显卡内存。
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                                stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)


        # 卷积核属性：3*3*channels   padding = 1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,    
                                stride=stride, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 卷积核属性：1*1*(channels*4)  padding = 0 以保持图片shape不变 维度变为原来的4倍
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1,    
                                stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)  #是否进行覆盖运算
        self.downsample = downsample
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

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
    num_classes:分类数
    '''
    def __init__(self,
                block: BigBlock,
                blocks_num: list,
                num_classes: int=10
                ):
        super(ResNet,self).__init__()
        self.in_channel = 64

        '''
        第一此卷积,使用7*7卷积核,步长为2,为了使shape缩小为原来的一半,padding为3,输入通道为RGB3通道,
        输出通道是64
        '''
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,
                             stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        #池化，缩小一半shape
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self.make_layer(block,64,blocks_num[0])
        self.layer2 = self.make_layer(block,128,blocks_num[1], stride=2)
        self.layer3 = self.make_layer(block,256,blocks_num[2], stride=2)
        self.layer4 = self.make_layer(block,512,blocks_num[3], stride=2)
     
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    # channel 残差模块第一层输入时的channel ,times 运行几次残差模块
    def make_layer(self,block,channel,times:int,stride=1):
        downsample = None   # 当在每个残差模块第一次卷积时需要改变channel
        '''
        有一点注意,第一个残差模块时只需要改变同等映射的channel一致(channel*4)就ok,而只后的残差模块还需要
        更改 高和宽,都要更改,所以添加stride参数。stride为1时表示不需要改变宽高,只改变channel,
        stride为2时表示宽高减半.channel减半.
        '''
        if self.in_channel != channel * block.expansion or stride!=1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,channel * block.expansion,1,stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layer = []
        layer.append(block(
            in_channel=self.in_channel,
            out_channel=channel,
            stride=stride,
            downsample=downsample
        ))
        self.in_channel = channel * block.expansion

        for i in range(1, times):
            layer.append(block(
            in_channel=self.in_channel,
            out_channel=channel,
        ))

        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  #展平
        x = self.fc(x)

        return x
def resnet50():
    return ResNet(BigBlock, [3, 4, 6, 3],num_classes=100)
def resnet101():
    return ResNet(BigBlock, [3, 4, 23, 3],num_classes=200)
print(resnet101())
