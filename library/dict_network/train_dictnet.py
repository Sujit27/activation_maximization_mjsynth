import sys
sys.path.append('../')

from dict_net import *
from dict_net_dataset import *
from helper_functions import *

from torch.utils.data  import SubsetRandomSampler
from torch import optim
import csv
import os
import glob
from statistics import mean
from pathlib import Path



def train_model(output_path,data_root,transform,prev_trained_checkpoint=None,num_labels=None,lr=0.005,batch_size=16,weight_decay=0.001,num_epochs=1):
    '''
    Given data location, creates a dictnet network, trains the model on the data and saves the model with dictionary of labels
    '''
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # create paths for checkpoint and best model
    checkpoint_path = os.path.join(output_path,'checkpoint.pth.tar')
    best_model_path = os.path.join(output_path,'model_best.pth.tar')

    # create dictnet dataset object
    dataset = DictNetDataset(data_root,num_labels)

    # save the label names for the dataset as a dictionary
    save_label_dict(dataset,output_path)

    # create network with number of output nodes same as number of distinct labels
    net = DictNet(num_labels)
    valid_loss_min = 1000000 #validation loss initialized with a high number, to start saving best model by comparing 

    # If provided, load model from a previous trained checkpoint
    if prev_trained_checkpoint is not None:
        checkpoint = torch.load(prev_trained_checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        valid_loss_min = checkpoint['valid_loss_min']

    net.to(device)

    # split training and validation data
    validation_split = 0.15
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    shuffle_dataset = True
    random_seed = 0

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # create a trainloader for the data subset
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=train_sampler,drop_last=True) 
    validationloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,sampler=valid_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    

    for epoch in range(num_epochs):
        train_loss_list = []
        valid_loss_list = []
        train_accuracy_list = []
        valid_accuracy_list = []

        # train
        net.train()
        for i, data in enumerate(trainloader, 0):
            images, targets = data

            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # moving average of training loss 
            #train_loss = train_loss + ( (1 / (i+1)) * (loss.data - train_loss))
            train_loss_list.append(loss.item())
            
            # training accuracy
            #outputs_exp = torch.exp(outputs)
            equality = (targets.data == outputs.max(dim=1)[1]) # compare predicted and ground truth labels
            accuracy = equality.type(torch.FloatTensor).mean()
            train_accuracy_list.append(accuracy.item())


        # validate
        net.eval()
        for j, data in enumerate(validationloader,0):
            images, targets = data

            images = images.to(device)
            targets = targets.to(device)
            
            outputs = net(images)

            loss = criterion(outputs, targets)

            # moving average of validation loss 
            #valid_loss = valid_loss + ( (1 / (i+1)) * (loss.data - valid_loss))
            valid_loss_list.append(loss.item())

            # validation accuracy
            #outputs_exp = torch.exp(outputs)
            equality = (targets.data == outputs.max(dim=1)[1]) # compare predicted and ground truth labels
            accuracy = equality.type(torch.FloatTensor).mean()
            valid_accuracy_list.append(accuracy.item())


        train_loss = mean(train_loss_list)
        valid_loss = mean(valid_loss_list)
        valid_accuracy = mean(valid_accuracy_list)
        train_accuracy = mean(train_accuracy_list)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.3f} \tValidation Accuracy: {:.3f}'.format(epoch,train_loss,valid_loss,train_accuracy,valid_accuracy))

        # create checkpoint
        checkpoint = {'epoch':epoch+1,'valid_accuracy_max':valid_accuracy,'valid_loss_min':valid_loss,'state_dict':net.state_dict()}

        save_ckp(checkpoint,False,checkpoint_path,best_model_path)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,valid_loss))

            save_ckp(checkpoint,True,checkpoint_path,best_model_path)
            valid_loss_min = valid_loss


