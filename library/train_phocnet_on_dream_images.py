import numpy as np
from statistics import mean
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data  import SubsetRandomSampler

import copy
from helper_functions import *
from phoc_network.phoc_dataset import *
from phoc_network import cosine_loss
from phoc_network.phoc_net import *
from phoc_network.predict_word_from_embd import *
#from dream_reader import *


   
def train_phocNet_on_dream_dataset(pooling_levels,word_length,training_data_path,output_path,
                                   num_epochs=100,lr=0.0001,batch_size=64,weight_decay=0.000,
                                   device=torch.device('cuda')):
    
    # create paths for checkpoint and best model
    checkpoint_path = os.path.join(output_path,'checkpoint.pth.tar')
    best_model_path = os.path.join(output_path,'model_best.pth.tar')
   
    train_data_set = PhocDataset(training_data_path,pooling_levels)
   
    # split training data into training and validation datasets
    validation_split = 0.15
    dataset_size = len(train_data_set)
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
    train_loader = DataLoader(train_data_set, batch_size=batch_size,sampler=train_sampler,drop_last=True,num_workers=0) 
    validation_loader = DataLoader(train_data_set, batch_size=batch_size,sampler=valid_sampler,num_workers=0)

    # create phocNet
    cnn = PHOCNet(train_data_set[0][1].shape[0],pooling_levels,input_channels=1,gpp_type='tpp')
    cnn.init_weights()
    criterion = nn.BCEWithLogitsLoss(size_average=True)
    
    cnn.to(device)
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr,
                                    weight_decay=weight_decay)
    valid_loss_min = 10000000 # validation loss min set to a high number for saving checkpoints

    for epoch in range(num_epochs):
        cnn.train()
        training_distance_list = []
        validation_distance_list = []
        training_loss_list = []
        validation_loss_list = []
        for i,data in enumerate(train_loader,0):
            imgs,embeddings,_,words = data

            imgs = imgs.to(device)
            embeddings = embeddings.to(device)

            optimizer.zero_grad()
            outputs = cnn(imgs)

            train_loss = criterion(outputs, embeddings) / batch_size
            train_loss.backward()
            optimizer.step()

            word_distance_array = find_string_distances(outputs.cpu().detach().numpy(),words,pooling_levels,word_length)
            edit_distance_error_avg = float(np.sum(word_distance_array)) / (batch_size)
            training_loss_list.append(train_loss.item())
            training_distance_list.append(edit_distance_error_avg)

        
        cnn.eval()
        for i,data in enumerate(validation_loader,0):
            imgs,embeddings,_,words = data

            imgs = imgs.to(device)
            embeddings = embeddings.to(device)

            outputs = cnn(imgs)
            validation_loss = criterion(outputs, embeddings) / batch_size
            word_distance_array = find_string_distances(outputs.cpu().detach().numpy(),words,pooling_levels,word_length)
            edit_distance_error_avg = float(np.sum(word_distance_array)) / (batch_size)
            validation_loss_list.append(validation_loss.item())
            validation_distance_list.append(edit_distance_error_avg)

        print("Epoch : {}, Training loss: {}, Training avg string distance : {}".format(epoch,mean(training_loss_list),mean(training_distance_list)))
        print("Epoch : {}, Validation loss: {}, Validation avg string distance: {}".format(epoch,mean(validation_loss_list),mean(validation_distance_list)))

        # create checkpoint
        checkpoint = {'epoch':epoch+1,'validation_loss_min':mean(validation_loss_list),'validation_distance_min':mean(validation_distance_list),'state_dict':cnn.state_dict(),'optimizer':optimizer.state_dict()}

        save_ckp(checkpoint,False,checkpoint_path,best_model_path)

        if mean(validation_loss_list) <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,mean(validation_loss_list)))

            save_ckp(checkpoint,True,checkpoint_path,best_model_path)
            valid_loss_min = mean(validation_loss_list)


        
