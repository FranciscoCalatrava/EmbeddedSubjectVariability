import torch
import torch.nn as nn
from collections import OrderedDict
import random
import numpy as np




class Classifier(nn.Module):
    def __init__(self, out_classes, seed, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_seed(seed)

        self.model = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(64,512)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(512,256)),
            ('bn2', nn.BatchNorm1d(256)),
            ('rel2', nn.ReLU()),
            ('linear3', nn.Linear(256,out_classes)),
            ('bn3', nn.BatchNorm1d(out_classes))
        ]))
        # self.model = nn.Sequential(OrderedDict([
        #     ('flatten', nn.Flatten()),
        #     ('linear1', nn.Linear(64,512)),
        #     ('dropout1', nn.Dropout(0.2)),
        #     ('bn1', nn.BatchNorm1d(512)),
        #     ('relu1', nn.ReLU()),
        #     ('linear2', nn.Linear(512,256)),
        #     ('dropout2', nn.Dropout(0.1)),
        #     ('bn2', nn.BatchNorm1d(256)),
        #     ('rel2', nn.ReLU()),
        #     ('linear3', nn.Linear(256,out_classes))
        #     # ('bn3', nn.BatchNorm1d(out_classes))
        # ]))

        # self.model = nn.Sequential(OrderedDict([
        #     ('flatten', nn.Flatten()),
        #     ('linear1', nn.Linear(64, 1024)),
        #     ('dropout1', nn.Dropout(0.3)),
        #     ('bn1', nn.BatchNorm1d(1024)),
        #     ('relu1', nn.LeakyReLU(0.1)),
        #     ('linear2', nn.Linear(1024, 512)),
        #     ('dropout2', nn.Dropout(0.3)),
        #     ('bn2', nn.BatchNorm1d(512)),
        #     ('relu2', nn.LeakyReLU(0.1)),
        #     ('linear3', nn.Linear(512, 256)),
        #     ('dropout3', nn.Dropout(0.2)),
        #     ('bn3', nn.BatchNorm1d(256)),
        #     ('relu3', nn.LeakyReLU(0.1)),
        #     ('linear4', nn.Linear(512, out_classes))
        #     ('bn3', nn.BatchNorm1d(out_classes))
        # ]))
        self._initialize_weights(seed)

    def _initialize_seed(self,seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _initialize_weights(self, seed):
        for m in self.model.modules():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        x = self.model(input)
        return x