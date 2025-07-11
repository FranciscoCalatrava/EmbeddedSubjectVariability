import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dropout_prob=0.3):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.leakyrelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Discriminator, self).__init__()
        self.disc1 = DiscriminatorBlock(in_channels, 32)
        self.disc2 = DiscriminatorBlock(32, 64)
        self.disc3 = DiscriminatorBlock(64, 128)
        self.fc1 = nn.Linear(128, 64)  # Assuming input size to fc1 is [256, 4, 4]
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.disc1(x)
        x = self.disc2(x)
        x = self.disc3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x