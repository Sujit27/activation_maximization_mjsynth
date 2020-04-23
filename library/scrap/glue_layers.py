import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from spectral import SpectralNorm
import numpy as np

class BasicGlueLayer(nn.Module):
    """A basic glue layer to take a dream as input and outputs a tensor of which can be then sent into the discriminator"""
    def __init__(self,inter_channels=8):
        super(BasicGlueLayer, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(1,inter_channels,kernel_size=3,padding=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU()
                )
        self.block2 = nn.Sequential(
                nn.Conv2d(inter_channels,1,kernel_size=3,padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU()
                )
    def forward(self,x):
        output = self.block1(x)
        output = self.block2(output)

        return output

