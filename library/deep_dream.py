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
from dict_net import *
from helper_functions import *

# Created by Sujit Sahoo, 13 Feb 2020
# sujit.sahoo@fau.de
   

class DeepDream():
    '''
    Given a network, input size to the network and channel wise mean,std of the data it was trained on,
    label specific 'deep dream' images can be created

    '''

    def __init__(self,net,input_size,data_mean=None,data_std=None,use_gaussian_filter=False):
        self.device = None
        self.net = net
        self.input_size = input_size
        self.data_mean = data_mean
        self.data_std = data_std
        self.input_2d = False
        self.input_3d = False
        self.ouputImage = None
        self.use_gaussian_filter = use_gaussian_filter
        # list variables used in randomDream method
        self.total_num_labels = self.net.final_layer.weight.shape[0]
        # set methods
        self.setDevice()
        self.setNetwork()
        self.check_input()
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
        
    def check_input(self):
        assert len(self.input_size) == 3
        if self.input_size[0] == 1: self.input_2d = True
        else: self.input_3d = True
        if self.input_2d:   
            if self.data_mean is None: 
                self.data_mean = 0.5
                print("Data means set at 0.5 by default")
            if self.data_std is None: 
                self.data_std = 0.5
                print("Data standard deviation set at 0.5 by default")


    def __call__(self,im=None,label=0,nItr=100,lr=0.1,random_seed=0):
        """Does activation maximization on a specific label for specified iterations,
           acts like a functor, and returns an image tensor
        """

        if im is None:
            im = self.createInputImage(random_seed)
            im = self.prepInputImage(im)
            im = im.to(self.device)

            im = Variable(im.unsqueeze(0),requires_grad=True)

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
        
        softmaxed_activation = F.softmax(self.net(im),dim=1)
        val,index = softmaxed_activation.max(1)
        print("Probablity after optimizing : {} and label {}".format(val[0],index[0]))

        #return im,val,index
        return im


    def createInputImage(self,random_seed):
        if self.input_2d:
            input_size = (self.input_size[1],self.input_size[2])
            #zeroImage_np = np.ones(input_size)*127
            np.random.seed(random_seed)
            zeroImage_np = np.random.random(input_size)*255
            zeroImage = Image.fromarray((zeroImage_np).astype('uint8'),'L')

        return zeroImage

    def prepInputImage(self,inputImage):
        if self.input_2d:
            if (self.data_mean is not None) and (self.data_std is not None):
                preprocess = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(self.data_mean,self.data_std),
                ])

        return preprocess(inputImage)

    def postProcess(self,image):
        image_tensor = torch.squeeze(image.data) # remove the batch dimension
        if self.input_2d:
            image_tensor = image_tensor*self.data_std[0] + self.data_mean[0] # std and mean for mjsynth 

            image_tensor = image_tensor.cpu() # back to host

            img = Image.fromarray((image_tensor.data.numpy()*255).astype('uint8'), 'L') #torch tensor to PIL image_tensor

        return img
    

    def show(self,img):
#         plt.figure(num=1, figsize=(12, 8), dpi=120, facecolor='w', edgecolor='k')
        plt.imshow(img,'gray')

    def save(self,image,fileName):
        #image = image.resize((32,256), Image.ANTIALIAS)
        image.save(fileName,'PNG')
        print('{} saved'.format(fileName))

    def setGaussianFilter(self,kernelSize=3,sigma=0.5):
        if self.input_2d:
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
#             gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

            pad = math.floor(kernelSize/2)

            gauss_filter = nn.Conv2d(in_channels=1, out_channels=1,padding=pad,
                                kernel_size=kernelSize, groups=1, bias=False)

            gauss_filter.weight.data = gaussian_kernel
            gauss_filter.weight.requires_grad = False
            self.gaussian_filter = gauss_filter.to(self.device)
            #print("gaussian_filter created")

class DeepDreamBatch(DeepDream):
    '''
    Works the same as DeepDream except that it creates dreams in batches
    '''
    def __init__(self,net,input_size,data_mean=None,data_std=None,use_gaussian_filter=False):
        super().__init__(net,input_size,data_mean,data_std,use_gaussian_filter)
        
    def random_batch_dream(self,batch_size,random_seed=0):
        """Does batch dreaming by randomly choosing n labels from range(num_labels) given by the batch_size with given random_seed
        """
        random.seed(random_seed)
        all_labels = [i for i in range(self.total_num_labels)]
        labels = random.sample(all_labels,batch_size)

        im = self.batch_dream(labels=labels,random_seed=random_seed)

        return im,labels
     
    def batch_dream(self,im=None,labels=[0,1,2,3],nItr=100,lr=0.1,random_seed=0):
        """Does activation maximization on list of labels for specified iterations,
           acts like a functor, and returns an image tensor
        """
        
        im = self.set_batch_dream(labels,random_seed)
        im = self.batch_dream_kernel(im,labels,nItr,lr)
        
        return im
        
    def set_batch_dream(self,labels,random_seed):
        
        im = self.createInputImage(random_seed)
        im = self.prepInputImage(im)
        im = torch.stack([im]*len(labels),dim=0)
        im = im.to(self.device)

        im = Variable(im,requires_grad=True)
        
        return im
        
        
    def batch_dream_kernel(self,im,labels,nItr,lr):
        
        for _ in range(nItr):

            out = self.net(im)
            
            loss = 0
            for i in range((out.shape)[0]):
                loss += out[i,labels[i]]
            
            loss.backward()

            avg_grad = np.abs(im.grad.data.cpu().numpy()).mean()
            norm_lr = lr / (avg_grad + 1e-20)
            im.data += norm_lr * im.grad.data
            im.data = torch.clamp(im.data,-1,1)
            
            if self.use_gaussian_filter == True:
                im.data = self.gaussian_filter(im.data)

            im.grad.data.zero_()
                

        return im
        
    
    def batch_postProcess(self,image_tensor):
        if self.input_2d:
            image_tensor = image_tensor*self.data_std[0] + self.data_mean[0] # std and mean for mjsynth 

            image_tensor = image_tensor.cpu() # back to host
        
        grid_img = utils.make_grid(image_tensor)
        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(np.transpose(grid_img.detach().numpy(),(1, 2, 0)))
        plt.savefig("batch_dream.png")

        return grid_img
    
    
class DeepDreamGAN(DeepDreamBatch):
    '''
    Given a network for dreaming, input size to the network and channel wise mean,std of the data it was trained on and another
    network to act as adversarial discriminator between real and dream,label specific 'deep dream' images can be created
    '''
    def __init__(self,net,input_size,data_mean=None,data_std=None,use_gaussian_filter=False,discrim_net=None):
        
        super().__init__(net,input_size,data_mean,data_std,use_gaussian_filter)
        self.discrim_net = discrim_net
        self.set_discrim_net()
        
    def set_discrim_net(self):
        if self.discrim_net is None: # if no discriminator is initialized, create one
            self.discrim_net = DictNet(2)
            print("Discriminator initialized with DictNet with 2 final outputs")
        else:
            num_final_output = self.discrim_net.final_layer.weight.shape[0]
            if num_final_output != 2:
                print("Error : Number of final output for the discminator is {}, it should be 2".format(num_final_output))
            else:
                print("Discriminator network set")

        self.discrim_net.to(self.device)
        
    def random_batch_dream_GAN(self,batch_size,random_seed=0):
        """Does batch dreaming by randomly choosing n labels from range(num_labels) given by the batch_size with given random_seed
        """
        random.seed(random_seed)
        all_labels = [i for i in range(self.total_num_labels)]
        labels = random.sample(all_labels,batch_size)

        im = self.batch_dream_GAN(labels=labels,random_seed=random_seed)

        return im,labels
        
        
    def batch_dream_GAN(self,labels=[0,1,2,3],n_adv_loops=10,nItr_g=100,nItr_d=10,lr_g=0.1,lr_d=0.1,random_seed=0):

        im = self.set_batch_dream(labels,random_seed)
        
        for _ in range(n_adv_loops):
            
            im = self.batch_dream_kernel(im,labels,nItr_g,lr_g)

            im = self.batch_discrim_kernel(im,labels,nItr_d,lr_d)


        return im
        
    def batch_discrim_kernel(self,im,labels,nItr,lr):
        
        for _ in range(nItr):

            out = self.net(im)
            
            loss = 0
            for i in range((out.shape)[0]):
                loss += out[i,1] # loss is the activation that how 'real' is the input image
            
            loss.backward()

            avg_grad = np.abs(im.grad.data.cpu().numpy()).mean()
            norm_lr = lr / (avg_grad + 1e-20)
            im.data += norm_lr * im.grad.data
            im.data = torch.clamp(im.data,-1,1)
            
            if self.use_gaussian_filter == True:
                im.data = self.gaussian_filter(im.data)

            im.grad.data.zero_()
                

        return im
                
    def train_discriminator(self,dataset_real,lr=0.001,batch_size=32,num_epochs=10,static_train=True):
        # the discriminator is trained with mixtur of real and dream images till training accuracy threshold is exceeded
        # if static_train is false, the dreams that are created during training for classification are adverserially created dreams. Takes musch more time

        trainloader = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size,shuffle=True) 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.discrim_net.parameters(), lr=lr)

        score_training_list = []
        for epoch in range(num_epochs):
            self.discrim_net.train()
            training_acc_score_list = []
            for i,data in enumerate(trainloader,0):
                real_images, _ = data # extracted a batch of real images and label
                random_seed = np.random.random()
                if static_train==True:
                    dream_images, _ = self.random_batch_dream(self,batch_size,random_seed=random_seed) # create dream images from dreamer
                else:
                    dream_images, _ = self.random_batch_dream_GAN(self,batch_size,random_seed=random_seed) # create dream images adversarially from dreamer and discriminator
                # set up the targets
                real_targets = torch.ones(batch_size)
                dream_targets = torch.zeros(batch_size)
                targets = torch.cat((real_targets,dream_targets))
                targets = targets.type(torch.LongTensor)

                # concat real and dream images
                images = torch.cat((real_images,dream_images),0)

                # move batch to device
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = self.discrim_net(images)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # measure training accuracy
                preds = one_hot_to_argmax(outputs)
                training_acc_score = skm.accuracy_score(targets.cpu().detach().numpy(),preds.cpu().detach().numpy())
                training_acc_score_list.append(training_acc_score)
        
            training_acc_avg = sum(training_acc_score_list)/len(training_acc_score_list)
            score_training_list.append(training_acc_avg)
            print("{},{}".format(epoch,training_acc_avg))

        output_file = os.path.join("../models/","discriminator.pth")
        torch.save(self.discrim_net.state_dict(),output_file)


 
        
def main():
    network = DictNet(5)
    dreamer = DeepDream(network,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
    dream = dreamer(label=1,nItr=500)  
    output = dreamer.postProcess(dream)
    dreamer.save(output,"dream_image.png") # saves image

if __name__ == "__main__":
    main()
