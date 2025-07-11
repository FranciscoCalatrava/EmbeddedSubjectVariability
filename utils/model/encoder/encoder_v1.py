
import torch
import torch.nn as nn
from collections import OrderedDict

class Multichannel_CNN(nn.Module):
    def __init__(self, window, no_of_class, no_of_sensor_channel):
        super().__init__()

        self.window = window
        self.number_of_classes = no_of_class
        self.number_of_sensors = no_of_sensor_channel
        
        self.section_1 = nn.Sequential(OrderedDict([
            ('Conv1_section1', nn.Conv2d(in_channels= 1, out_channels=50, kernel_size= (1,5))),
            # ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            ('relu1_section1', nn.ReLU()),
            ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            ('maxpool1_section1', nn.MaxPool2d(kernel_size=(1,2)))
            # ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1))
        ]))
        self.section_2 = nn.Sequential(OrderedDict([
            ('Conv1_section2', nn.Conv2d(in_channels= 50, out_channels=40, kernel_size= (1,5))),
            # ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            ('relu1_section2', nn.ReLU()),
            ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            ('maxpool1_section2', nn.MaxPool2d(kernel_size=(1,2)))
            # ('normalization1_section2', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1))
        ]))
        self.section_3 = nn.Sequential(OrderedDict([
            ('Conv1_section3', nn.Conv2d(in_channels= 40, out_channels=20, kernel_size= (1,3))),
            # ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            ('relu1_section3', nn.ReLU()),
            ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            # ('normalization1_section3', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k=1))
        ]))
        self.section_4 = nn.Sequential(OrderedDict([
            ('Conv1_section4', nn.Conv2d(in_channels= 20, out_channels=400, kernel_size= (self.number_of_sensors, 1))),
            ('tanh1_section4', nn.Tanh()),
            ('relu1_section4', nn.ReLU()),
            ('normalization1_section1', nn.LocalResponseNorm(size = 5, alpha= 2e-4, beta = 0.75, k =1)),
            ('adaptative_average_poopling_section_4', nn.AdaptiveAvgPool2d((1,1)))

        ]))
        # self.section_5 = nn.Sequential(OrderedDict([
        #     ('fully_connected_section5', nn.Linear(400, self.number_of_classes))
        # ]))

        self.flatten = nn.Flatten()
    def forward(self, x):
        # print(x.shape)
        # x = torch.transpose(x,2,1)
        # print(x.shape)
        x = torch.unsqueeze(x, dim= 1)
        x = self.section_1(x)
        x = self.section_2(x)
        x = self.section_3(x)
        # print(x.shape)
        x = torch.squeeze(torch.squeeze(self.section_4(x)))
        # print(x.shape)
        # x = self.section_5(x)
        return x,x
    