import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
from torch.nn import functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import scipy
import scipy.misc
from scipy import ndimage
import math
import os
import random

# Created by Sujit Sahoo, 26 Sept 2019
# sujit.sahoo@fau.de

    

class DeepDream():
    '''
    Given a label (number between 0 and 1000) of the ImageNet and
    an input image(zero image by default),label specific 'deep dream'
    images can be created

    '''

    def __init__(self,net,use_gaussian_filter=False):
        self.device = None
        self.net = net
        self.ouputImage = None
        self.use_gaussian_filter = use_gaussian_filter
        # list variables used in randomDream method
        self.labels = [i for i in range(1000)]
        # set methods
        self.setDevice()
        self.setNetwork()
        if self.use_gaussian_filter == True:
            print("Gaussian filter will be used")
            self.gaussian_filter = None
            self.setGaussianFilter()

    def setDevice(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used to run this program: ",self.device)


    def setNetwork(self):
        print("Loading the network...")
        
        self.net.eval() # inference mode

        self.net.to(self.device)
        print("Network Loaded")

#    def __call__(self,im=None,label=0,nItr=100,lr=0.1):
#        """Does activation maximization on a specific label for specified iterations,
#           acts like a functor, and returns an image tensor
#        """
#
#        if im is None:
#            im = self.createInputImage()
#            im = self.prepInputImage(im)
#            im = im.to(self.device)
#
#            im = Variable(im.unsqueeze(0),requires_grad=True)
#
#            # offset by the min value to 0 (shift data to positive)
#            min_val = torch.min(im.data)
#            im.data = im.data - min_val
#
#        print("Dreaming...")
#
#        for i in range(nItr):
#
#            optimizer = torch.optim.SGD([im],lr)
#            out = self.net(im)
#            loss = -out[0,label]
#
#            loss.backward()
#            optimizer.step()
#            
#            if self.use_gaussian_filter == True:
#                im.data = self.gaussian_filter(im.data)
#
#            im.grad.data.zero_()
#
#        return im

    def __call__(self,im=None,label=0,nItr=100,lr=0.1):
        """Does activation maximization on a specific label for specified iterations,
           acts like a functor, and returns an image tensor
        """

        if im is None:
            im = self.createInputImage()
            im = self.prepInputImage(im)
            im = im.to(self.device)

            im = Variable(im.unsqueeze(0),requires_grad=True)

            # offset by the min value to 0 (shift data to positive)
#            min_val = torch.min(im.data)
#            im.data = im.data - min_val

        print("Activation before optimizing : {}".format(self.net(im)[0,label]))
        softmaxed_activation = F.softmax(self.net(im),dim=1)
        val,index = softmaxed_activation.max(1)
        print("Probablity before optimizing : {} and label {}".format(val[0],index[0]))
        print("Dreaming...")

        for i in range(nItr):

            out = self.net(im)
            #loss = -out[0,label]
            loss = out[0,label]
            loss.backward()

            avg_grad = np.abs(im.grad.data.cpu().numpy()).mean()
            norm_lr = lr / (avg_grad + 1e-20)
            im.data += norm_lr * im.grad.data
            im.data = torch.clamp(im.data,-1,1)
            
            if self.use_gaussian_filter == True:
                im.data = self.gaussian_filter(im.data)

            im.grad.data.zero_()
        
        print("Activation after optimizing : {}".format(self.net(im)[0,label]))
        softmaxed_activation = F.softmax(self.net(im),dim=1)
        val,index = softmaxed_activation.max(1)
        print("Probablity after optimizing : {} and label {}".format(val[0],index[0]))

        return im

    def randomDream(self,im=None,randomSeed=0):
        """Does activation maximization on a random label for randomly chosen learning rate,number of iterations and gaussian filter size, and returns an image tensor
        """
        random.seed(randomSeed)
        rand_nItr = np.asscalar(np.random.normal(500,40,1).astype(int))
        rand_lr = np.asscalar(np.random.normal(0.12,0.01,1))
        rand_label = random.choice(self.labels)
        if self.use_gaussian_filter == True:
            rand_sigma = np.asscalar(np.random.normal(0.45,0.05,1))
            self.setGaussianFilter(sigma=rand_sigma)

        im = self.__call__(im,label=rand_label,nItr=rand_nItr,lr=rand_lr)

        return im


    def createInputImage(self):
        zeroImage_np = np.ones((32,256))*127
        zeroImage = Image.fromarray((zeroImage_np).astype('uint8'),'L')

        return zeroImage

    def prepInputImage(self,inputImage):
        preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.47,),(0.14,)),
        ])

        return preprocess(inputImage)

    def postProcess(self,image):
        image_tensor = torch.squeeze(image.data) # remove the batch dimension

#        image_tensor.transpose_(0,1) # convert from CxHxW to HxWxC format
#        image_tensor.transpose_(1,2)
        image_tensor = image_tensor*0.14 + 0.47 # std and mean for mjsynth 
  
        image_tensor = image_tensor.cpu() # back to host
  
#        # TRUNCATE TO THROW OFF DATA OUTSIDE 5 SIGMA
#        mean = torch.mean(image_tensor)
#        std = torch.std(image_tensor)
#        upper_limit = mean + 5 * std
#        lower_limit = mean - 5 * std
#        image_tensor.data = torch.clamp_(image_tensor.data,lower_limit,upper_limit)
#
#
#        # normalize data to lie between 0 and 1
#        image_tensor.data = (image_tensor.data - lower_limit) / (10*std)
#
        img = Image.fromarray((image_tensor.data.numpy()*255).astype('uint8'), 'L') #torch tensor to PIL image_tensor

        return img

    def show(self):
        plt.figure(num=1, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
        plt.imshow(np.asarray(self.outputImage))

    def save(self,image,fileName):
        #image = image.resize((32,256), Image.ANTIALIAS)
        image.save(fileName,'PNG')
        print(f'{fileName} saved')



    def setGaussianFilter(self,kernelSize=3,sigma=0.45):

        # Create a x, y coordinate grid of shape (kernelSize, kernelSize, 2)
        x_cord = torch.arange(kernelSize)
        x_grid = x_cord.repeat(kernelSize).view(kernelSize, kernelSize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.float()

        mean = (kernelSize - 1)/2.
        variance = sigma**2.


        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /(2*variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernelSize, kernelSize)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        pad = math.floor(kernelSize/2)

        gauss_filter = nn.Conv2d(in_channels=3, out_channels=3,padding=pad,
                            kernel_size=kernelSize, groups=3, bias=False)

        gauss_filter.weight.data = gaussian_kernel
        gauss_filter.weight.requires_grad = False
        self.gaussian_filter = gauss_filter.to(self.device)
        #print("gaussian_filter created")




if __name__ == "__main__":
    dreamer = DeepDream()
    dreamer.setGaussianFilter(3,0.48) # optional step
    dreamtImage = dreamer(label=130)  # 130 is the label for flamingo, see https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    dreamtImage =  dreamer.postProcess(dreamtImage)
    dreamer.show() # shows image
    dreamer.save(dreamtImage,"myImage.png") # saves image

