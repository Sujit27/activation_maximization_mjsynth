import torch
import numpy as np
import os
import dagtasets as dg
import torch.nn as nn
from torch.utils import data
from collections import Counter
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import sklearn.metrics as skm
import re

# Convolutional neural network 
class DictNet(nn.Module):
    def __init__(self, num_classes=772, conv_capacity=16, fc_capacity=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_capacity*4 , kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv_capacity*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_capacity*4, conv_capacity*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv_capacity*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_capacity*8, conv_capacity*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_capacity*16),
            nn.ReLU())
            
        self.conv4 = nn.Sequential(
            nn.Conv2d(conv_capacity*16, conv_capacity*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_capacity*32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            
        self.conv5 = nn.Sequential(
            nn.Conv2d(conv_capacity*32, conv_capacity*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_capacity*32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(
            nn.Linear(2*16*conv_capacity*32, fc_capacity*32),
            nn.BatchNorm1d(fc_capacity*32),
            nn.ReLU())
    
        self.fc2 = nn.Sequential(
            nn.Linear(fc_capacity*32, fc_capacity*32),
            nn.BatchNorm1d(fc_capacity*32),
            nn.ReLU())
        
        self.final = nn.Linear(fc_capacity*32, num_classes)
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.final(out)
        #out = self.softmax(out)
        return out

class DictNet2(DictNet):
    def __init__(self, num_classes=772, conv_capacity=16, fc_capacity=128):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2*8*conv_capacity*32, fc_capacity*32),
            nn.BatchNorm1d(fc_capacity*32),
            nn.ReLU())
       


