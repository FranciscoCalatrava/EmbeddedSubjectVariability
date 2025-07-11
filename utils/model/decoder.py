import torch.nn as nn
import torch
import random
import numpy as np

class Decoder(nn.Module):
    def __init__(self, input_shape, num_blocks=[], out_channel = 18,seed = 0):
        super(Decoder, self).__init__()
        self._initialize_seed(seed)
        self.in_channels, _, _ = input_shape
        self.in_channels = 128
        self.linear1 = nn.Linear(64, 128 * 1 * 1) 
        self.relu1 = nn.ReLU()
        parameters = [128,128,(1,3), (1,1), (0,1), 16, (0,1)]
        self.block_1 = BasicBlock_1D_with_upsampling(128,128, parameters, (1,1))
        self.block_2 = BasicBlock_1D(128, 64, stride = (1,1), downsample = False)
        parameters = [64,64,(1,3), (1,1), (0,1), 16, (0,2)]
        self.block_3 = BasicBlock_1D_with_upsampling(64,64, parameters, (1,1))
        self.block_4 = BasicBlock_1D(64, 64, stride = (1,1), downsample = False)
        self.conv1 = nn.ConvTranspose2d(64, out_channel, kernel_size=(1, 5), stride=(1, 2), padding=(0,2), dilation = 1, bias=False,  output_padding= (0,1))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.01)
        self.conv2 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(1, 5), stride=(1, 2), padding=(0,2), dilation = 1, bias=False,  output_padding= (0,1))
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(0.01)
        self.conv3 = nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(1, 5), stride=(1, 2), padding=(0,2), dilation = 1, bias=False,  output_padding= (0,1))
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.LeakyReLU(0.01)

        self._initialize_weights()

    def _initialize_seed(self,seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = x.view(-1, 128, 1, 1)  # Reshape to match the encoder's last layer output
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x
    
class BasicBlock_1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), stride=stride, padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out
class BasicBlock_1D_with_upsampling(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, parameters, stride=1, downsample=None):
        super(BasicBlock_1D_with_upsampling, self).__init__()
        self.conv1 = nn.ConvTranspose2d(parameters[0], parameters[1], kernel_size=parameters[2], stride=parameters[3], padding=parameters[4], dilation = parameters[5], bias=False, output_padding= parameters[6])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

        
        