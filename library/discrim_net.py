import torch
import numpy as np
import torch.nn as nn

class DiscrimNet(nn.Module):
    # DictNet for images with size 32x128, padding transform
    def __init__(self, num_classes=2, conv_capacity=16, fc_capacity=128):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(1, conv_capacity*4 , kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv_capacity*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
            

        self.fc_layers = nn.Sequential(
            # 1st fc layer
            nn.Linear(16*8*conv_capacity*32, fc_capacity*32),
            nn.BatchNorm1d(fc_capacity*32),
            nn.ReLU())

        
        self.final_layer = nn.Linear(fc_capacity*32, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layers(out)
        out = self.final_layer(out)
        out = self.softmax(out)
        return out
 
