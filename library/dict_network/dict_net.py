import torch
import numpy as np
import torch.nn as nn

def grow_net(input_net,output_net_num_labels):
    # create a new DictNet with number of output labels increased and return the output_network
    output_net = DictNet(output_net_num_labels)
    
    #for i in range(len(input_net.cnn_layers)):
    output_net.cnn_layers.load_state_dict(input_net.cnn_layers.state_dict())
    
    #for i in range(len(input_net.fc_layers)):
    output_net.fc_layers.load_state_dict(input_net.fc_layers.state_dict())

        
    input_net_num_labels = input_net.final_layer.weight.data.shape[0]
    output_net.final_layer.weight.data[:input_net_num_labels,:] = input_net.final_layer.weight.data
    output_net.final_layer.bias.data[:input_net_num_labels] = input_net.final_layer.bias.data
    
    return output_net



# Convolutional neural network 
class DictNet(nn.Module):
    # DictNet for images with size 32x128, padding transform
    def __init__(self, num_classes=772, conv_capacity=16, fc_capacity=128):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(1, conv_capacity*4 , kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv_capacity*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 2nd conv layer
            nn.Conv2d(conv_capacity*4, conv_capacity*8 , kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv_capacity*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3rd conv layer
            nn.Conv2d(conv_capacity*8, conv_capacity*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_capacity*16),
            nn.ReLU(),
            # 4th conv layer
            nn.Conv2d(conv_capacity*16, conv_capacity*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_capacity*32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #5th conv layer
            nn.Conv2d(conv_capacity*32, conv_capacity*32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(conv_capacity*32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layers = nn.Sequential(
            # 1st fc layer
            nn.Linear(2*8*conv_capacity*32, fc_capacity*32),
            nn.BatchNorm1d(fc_capacity*32),
            nn.ReLU())
            # 2nd fc layer
#            nn.Linear(fc_capacity*32, fc_capacity*32),
#            nn.BatchNorm1d(fc_capacity*32),
#            nn.ReLU())
        
        self.final_layer = nn.Linear(fc_capacity*32, num_classes)
        #self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc_layers(out)
        out = self.final_layer(out)
        #out = self.softmax(out)
        return out

# Convolutional neural network 

