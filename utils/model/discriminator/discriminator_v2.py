import torch.nn as nn
from collections import OrderedDict
import torch

class Discriminator_1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(OrderedDict([
            ('Gradient_Reversal', GradientReversal(alpha=0.60)),
            ('convmax1', self.discriminator_block(1,32,9)),
            ('convmax2', self.discriminator_block(32,64,5)),
            ('convmax3', self.discriminator_block_dropout(64,128,3)),
            ('convmax4', self.discriminator_block_dropout(128,256,3)),
            ('Flatten', nn.Flatten()),
            ('linear1', nn.Linear(7680,256)),
            ('relu1', nn.ReLU()),
            ('Dropout', nn.Dropout(0.2)),
            ('linear2', nn.Linear(256,64)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(64,2))
            
        ]))
    def discriminator_block(self, channel_in, channel_out, kernel):
        disc_block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(channel_in, channel_out, kernel, stride= 2)),
            ('bn1', nn.BatchNorm1d(channel_out)),
            ('relu', nn.ReLU())
        ]))
        return disc_block
    
    def discriminator_block_dropout(self, channel_in, channel_out, kernel):
        disc_block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(channel_in, channel_out, kernel, stride= 2)),
            ('bn1', nn.BatchNorm1d(channel_out)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.3))
        ]))
        return disc_block

    def forward(self, input):
        x = torch.unsqueeze(input, 1)
        x = self.model(x)
        return x


import torch
from torch import nn

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
    

from torch.autograd import Function

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None
revgrad = GradientReversal.apply