import torch
import numpy as np
import torch.nn as nn

class DiscrimNet(nn.Module):
    # DictNet for images with size 32x256, padding transform
    def __init__(self, num_classes=772, conv_capacity=16, fc_capacity=128):
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
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layers(out)
        out = self.final_layer(out)
        #out = self.softmax(out)
        return out
 
#Discriminator for SRGAN with 1 channel input 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))

       


