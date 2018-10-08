import torch
import torch.nn as nn
import torchvision

class ResBlock(nn.Module):
    def __init__(self, input_channel, channel_list = [64, 64, 256], is_first=False):
        super(ResBlock ,self).__init__()
        self.channel_list = channel_list
        self.input_channel = input_channel
        self.is_first = is_first
        self.relu = nn.ReLU(inplace=True)
        self.conv1_reduce = nn.Conv2d(input_channel, channel_list[0], 1, stride=1, padding=0)
        self.bn1_reduce = nn.BatchNorm2d(channel_list[0])
        self.conv1 = nn.Conv2d(channel_list[0], channel_list[1], 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_list[1])
        self.conv1_increase = nn.Conv2d(channel_list[1], channel_list[2], 1, stride=1, padding=0)
        self.bn1_increase = nn.BatchNorm2d(channel_list[2])
        if self.is_first:
            self.conv1_proj = nn.Conv2d(input_channel, channel_list[2], 1, stride=1, padding=0)
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
            self.module_list.append(ResBlock(input_channel=channel_list[2], channel_list=channel_list))
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
        self.conv1_1 = nn.Conv2d(self.channel, 64, 3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.conv1_3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2d(128)
        self.pooling1 = nn.MaxPool2d(3, stride=2, padding=1)
        


class DORN(nn.Module):
    def __init__(self, width, height):
        super(DORN, self).__init__()
        self.width = width
        self.height = height

