import torch
import torch.nn as nn
import torchvision

class ResBlock(nn.Module):
    def __init__(self, input_channel, \
            channel_list = [64, 64, 256], is_first=False):
        super(ResBlock ,self).__init__()
        self.channel_list = channel_list
        self.input_channel = input_channel
        self.is_first = is_first
        self.relu = nn.ReLU(inplace=True)
        self.conv1_reduce = nn.Conv2d(input_channel, channel_list[0], \
                1, stride=1, padding=0, bias=False)
        self.bn1_reduce = nn.BatchNorm2d(channel_list[0])
        self.conv1 = nn.Conv2d(channel_list[0], channel_list[1], 3, \
                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_list[1])
        self.conv1_increase = nn.Conv2d(channel_list[1], channel_list[2], \
                1, stride=1, padding=0, bias=False)
        self.bn1_increase = nn.BatchNorm2d(channel_list[2])
        if self.is_first:
            self.conv1_proj = nn.Conv2d(input_channel, channel_list[2], \
                    1, stride=1, padding=0, bias=False)
            self.bn1_proj = nn.BatchNorm2d(channel_list[2])
    
    def forward(self, x):
        x1 = self.relu(self.bn1_reduce(self.conv1_reduce(x)))
        x1 = self.relu(self.bn1(self.conv1(x1)))
        x1 = self.bn1_increase(self.conv1_increase(x1))
        if self.is_first:
            x2 = self.bn1_proj(self.conv1_proj(x))
            out = x1 + x2
        else:
            out = x1 + x
        out = self.relu(out)
        return out

class ResBlockCollection(nn.Module):
    def __init__(self, input_channel, channel_list, block_num):
        super(ResBlockCollection, self).__init__()
        self.input_channel = input_channel
        self.channel_list = channel_list
        self.block_num = block_num
        self.module_list = nn.ModuleList()
        self.module_list.append(ResBlock(input_channel=input_channel, channel_list=channel_list, is_first=True))
        for i in range(1, self.block_num):
            self.module_list.append(ResBlock(input_channel=channel_list[2],\
                    channel_list=channel_list))
        self.seq = nn.Sequential()
        i = 0
        for module in self.module_list:
            self.seq.add_module(f"block{i}", module)
            i += 1
    
    def forward(self, x):
        return self.seq(x)

class DenseFeatureExtractor(nn.Module):

    def __init__(self, channel):
        super(DenseFeatureExtractor, self).__init__()
        self.channel = channel
        self.conv1_1 = nn.Conv2d(self.channel, 64, 3, stride=2, \
                padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(128)
        self.pooling1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.blockcolle1 = ResBlockCollection(input_channel=128, \
                channel_list=[64, 64, 256], block_num=3)
        self.blockcolle2 = ResBlockCollection(input_channel=256, \
                channel_list=[128, 128, 512], block_num=4)
        self.blockcolle3 = ResBlockCollection(input_channel=512, \
                channel_list=[256, 256, 1024], block_num=23)
        self.blockcolle4 = ResBlockCollection(input_channel=1024, \
                channel_list=[512, 512, 2048], block_num=3)

class FullImageEncoder(nn.Module):
    def __init__(self):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(16, stride=16)
        self.dropout = nn.Dropout2d(p=0.5)
        self.global_fc = nn.Linear(2048, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.upsample = nn.UpsamplingBilinear2d(size=(49, 65))

    def forward(self, x):
        x1 = self.global_pooling(x)
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048)
        x4 = self.relu(self.global_fc(x3))
        x5 = self.conv1(x4)
        out = self.upsample(x5)
        return out

class SceneUnderstandingModule(nn.Module):
    def __init__(self):
        super(SceneUnderstandingModule, self).__init__()
        self.encoder = FullImageEncoder()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=12, dilation=12),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=18, dilation=18),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512*5, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, 142, 1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        x6 = torch.cat((x1, x2, x3, x4, x5), 0)
        out = self.concat_process(x6)
        return out

class DORN(nn.Module):
    def __init__(self, width, height, channel):
        super(DORN, self).__init__()
        self.width = width
        self.height = height
        self.channel = channel
        self.feature_extractor = DenseFeatureExtractor(self.channel)

