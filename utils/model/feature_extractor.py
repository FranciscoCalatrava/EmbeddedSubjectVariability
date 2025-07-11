import torch
import torch.nn as nn
from collections import OrderedDict
import random
import numpy as np



class Feature_Extractor(nn.Module):
    def __init__(self, input_shape , num_blocks = [], in_channel= 18, seed = 0):
        super(Feature_Extractor, self).__init__()
        self._initialize_seed(seed)
        in_channels, height, width = input_shape

        self.in_channels = in_channels
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0,2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.001)
        self.maxpool = nn.MaxPool2d(kernel_size = (1,3), stride=(1,2), padding=(0,1), dilation=1, return_indices=False, ceil_mode=False)

        self.layer1 = self._make_layer_1D(64, num_blocks[0],  stride=1)
        self.layer2 = self._make_layer_1D(128, num_blocks[1], stride=(1,2))


        self.adpavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,64)
        self.relu_1 = nn.ReLU()
        

        self._initialize_weights()

    def _initialize_seed(self,seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)


    def _make_layer_1D(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = nn.ModuleList()
        layers.append(BasicBlock_1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock_1D.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock_1D(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # print(x.shape)
        x = self.adpavgpool(x)
        x = self.flatten(x)
        x = self.relu_1(self.linear1(x))
        return x,x

class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.LeakyReLU(0.001)
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
