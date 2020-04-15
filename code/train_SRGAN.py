#!/usr/bin/env python

import sys
sys.path.append("../library/SR_GAN")
sys.path.append("../library")
import argparse
import os
from pathlib import Path
import random

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#from tensorboard_logger import configure, log_value

import dagtasets as dg
from real_dream_dataset import *
from dict_net import *
from models import Generator, Discriminator
#from utils import Visualizer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--network_model', type=str, default='../models2/net_1000_0.001_200_0.0.pth', help='Trained classifier network')
parser.add_argument('--model_save_loc', type=str, default='.', help='location where the generator checkpoints are saved')
parser.add_argument('--real_dataroot', type=str, default='/var/tmp/on63ilaw/mjsynth', help='path to real images dataset')
parser.add_argument('--dream_dataroot', type=str, default='/var/tmp/on63ilaw/mjsynth/sample_dreams_dataset', help='path to dream images dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--disp', type=int, default=50, help='number of iterations for display of losses')
parser.add_argument('--generatorLR', type=float, default=0.00001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.00001, help='learning rate for discriminator')
parser.add_argument('--gen_loss_ratio', type=float, default=0.7, help='ratio of two different types of losses for the generator, should be less than one ')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')

opt = parser.parse_args()
#print(opt)

Path("./out").mkdir(parents=True,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_dream_acc_loss(network,input_images,target_labels,loss_criterion):
    output = network(input_images)
    loss = loss_criterion(output,target_labels)

    return loss

def main():

    dataset = RealDreamDataset(opt.real_dataroot,opt.dream_dataroot)

    num_labels = int(((os.path.basename(opt.network_model)).split("_"))[1])
    network = DictNet(num_labels)
    network.load_state_dict(torch.load(opt.network_model))
    network.eval()
    network.to(device)
    loss_criterion = nn.CrossEntropyLoss()

    # create dataloaders
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=opt.batchSize,
            shuffle=True,num_workers=int(opt.workers))

    # create generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    #print(generator)
    #print(discriminator)

    # Define Loss function
    adversarial_criterion = nn.CrossEntropyLoss()

    # targets for dream and real images
    zeros_const = torch.zeros(opt.batchSize,dtype=torch.long)
    ones_const = torch.ones(opt.batchSize,dtype=torch.long)

    # move tensors to cuda
    generator.to(device)
    discriminator.to(device)
    zeros_const = zeros_const.to(device)
    ones_const = ones_const.to(device)

    # Define optimizers
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)


    ####################################
    # SRGAN training
    print("Generator adv loss, Generator dream acc loss, Discriminator adv loss")
    for epoch in range(opt.nEpochs):

        for i, data in enumerate(data_loader):

            # extract images
            real_images,dream_images,dream_labels= data

            real_images = real_images.to(device)
            dream_images = dream_images.to(device)
            dream_labels = dream_labels.to(device)

            # pass the dream images through generator
            dream_images_generated = generator(dream_images)

           
            ######### Train discriminator #########
            discriminator.zero_grad()

            real_output = discriminator(real_images)
            dream_output = discriminator(dream_images_generated)

            discriminator_loss = (adversarial_criterion(real_output, ones_const) + adversarial_criterion(dream_output, zeros_const))/opt.batchSize
            
            discriminator_loss.backward(retain_graph=True)
            optim_discriminator.step()

            ######### Train generator #########
            generator_dream_loss = find_dream_acc_loss(network,dream_images_generated,dream_labels,loss_criterion)/opt.batchSize
            generator.zero_grad()
            generator_adv_loss = (adversarial_criterion(dream_output, ones_const))/opt.batchSize

            generator_loss = opt.gen_loss_ratio * generator_dream_loss + (1-opt.gen_loss_ratio) * generator_adv_loss
            
            generator_loss.backward()
            optim_generator.step()  

            if i % opt.disp == 0:
                print("{},{},{}".format(generator_adv_loss.item(),generator_dream_loss.item(),discriminator_loss.item())) 
        # Do checkpointing
        if epoch % 5 == 4:
            generator_checkpoint = 'generator_checkpoint_' + str(epoch) + '.pth'
            torch.save(generator.state_dict(), os.path.join(opt.model_save_loc,generator_checkpoint))
    torch.save(discriminator.state_dict(), 'out/discriminator_final.pth')

if __name__ == "__main__":
    main()
