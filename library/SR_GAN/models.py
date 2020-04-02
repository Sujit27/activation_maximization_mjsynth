# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802
TODO:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def swish(x):
    return x * F.sigmoid(x)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(y)) + x)

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)

    def forward(self, x):
        return swish(self.shuffler(self.conv(x)))

class Generator(nn.Module):
    def __init__(self, n_residual_blocks=8):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i+1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #self.upsample = upsampleBlock(64, 128)

        self.conv3 = nn.Conv2d(64, 1, 9, stride=1, padding=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i+1))(y)

        x = F.relu(self.bn2(self.conv2(y)) + x)

        #x = self.upsample(x)

        return F.relu(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
#        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
#        self.bn4 = nn.BatchNorm2d(128)
#        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#        self.bn5 = nn.BatchNorm2d(256)
#        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
#        self.bn6 = nn.BatchNorm2d(256)
#        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#        self.bn7 = nn.BatchNorm2d(512)
#        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
#        self.bn8 = nn.BatchNorm2d(512)
#
#        # Replaced original paper FC layers with FCN
#        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
        self.fc1 = nn.Linear(64*32*8,256)
        self.bn4 = nn.BatchNorm1d(256)

        self.final = nn.Linear(256,2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1,64*32*8)
        
        x = F.relu(self.bn4(self.fc1(x)))
        return F.softmax(self.final(x),dim=1)
