#!/usr/bin/env python

import argparse
import os
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

import dagtasets as dg

from models import Generator, Discriminator
from utils import Visualizer

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--real_dataroot', type=str, default='/var/tmp/on63ilaw/mjsynth', help='path to real images dataset')
parser.add_argument('--dream_dataroot', type=str, default='/var/tmp/on63ilaw/mjsynth/sample_dreams_dataset', help='path to dream images dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.00001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.00001, help='learning rate for discriminator')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = dg.mjsynth.mjsynth_gray_scale

real_dataset = dg.mjsynth.MjSynthWS(opt.real_dataroot,transform)
dream_dataset = dg.mjsynth.MjSynthWS(opt.dream_dataroot,transform)


real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dream_dataloader = torch.utils.data.DataLoader(dream_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


# create generator and discriminator
generator = Generator(8, opt.upSampling)
discriminator = Discriminator()

# Define Loss function
adversarial_criterion = nn.BCELoss()

# targets for dream and real images
zeros_const = torch.zeros(opt.batchSize,dtype=torch.long)
ones_const = torch.ones(opt.batchSize,dtype=torch.long)

# move tensors to cuda
generator.to(device)
discriminator.to(device)
zeros_const.to(device)
ones_const.to(device)

# Define optimizers
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

#configure('logs/' + opt.dataset + '-' + str(opt.batchSize) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR), flush_secs=5)
#visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)


# SRGAN training
print 'SRGAN training'
for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, data in enumerate(dataloader):
        # Generate data
        high_res_real, _ = data

        # Downsample images to low resolution
        for j in range(opt.batchSize):
            low_res[j] = scale(high_res_real[j])
            high_res_real[j] = normalize(high_res_real[j])

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(low_res).cuda())
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)
        
        ######### Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data[0]
        
        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data[0]
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data[0]
        
        generator_total_loss.backward()
        optim_generator.step()   
        
        ######### Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (epoch, opt.nEpochs, i, len(dataloader),
        discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))
        visualizer.show(low_res, high_res_real.cpu().data, high_res_fake.cpu().data)

    sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
    mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
    mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))

    log_value('generator_content_loss', mean_generator_content_loss/len(dataloader), epoch)
    log_value('generator_adversarial_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
    log_value('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
    log_value('discriminator_loss', mean_discriminator_loss/len(dataloader), epoch)

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
    torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)

# Avoid closing
while True:
    pass
