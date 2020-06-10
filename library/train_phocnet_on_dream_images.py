import numpy as np
from sklearn.metrics import accuracy_score
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data  import SubsetRandomSampler

import copy
from phoc_network.phoc_dataset import *
from phoc_network import cosine_loss
from phoc_network.phoc_net import *
from phoc_network.predict_word_from_embd import *
#from dream_reader import *


def evaluate_cnn(dream_reader,output,words):
    indices_pred,words_pred = dream_reader.read_from_network_output(output)
    indices_target = dream_reader.convert_words_to_indices(words)

    acc_score = accuracy_score(indices_target,indices_pred)

    return acc_score
    
def train_phocNet_on_dream_dataset(pooling_levels,word_length,training_data_path,test_data_path=None,
                                   num_epochs=100,lr=0.0001,batch_size=64,weight_decay=0.000,
                                   device=torch.device('cuda')):
    
    train_data_set = PhocDataset(training_data_path,pooling_levels)
    train_loader = DataLoader(train_data_set,batch_size=batch_size,shuffle=True,num_workers=0)
   
    if test_data_path is not None:
        test_data_set = PhocDataset(test_data_path,pooling_levels)
        test_loader = DataLoader(test_data_set,batch_size=batch_size,shuffle=True,num_workers=0)
    
#    with open(lex_txt_file) as f:
#        lex_list = f.readlines()
#    lex_list = [word[:-1] for word in lex_list] 
    #dream_reader = DreamReader(lex_list,pooling_levels)
    
    cnn = PHOCNet(train_data_set[0][1].shape[0],pooling_levels,input_channels=1,gpp_type='tpp')
    cnn.init_weights()
    criterion = nn.BCEWithLogitsLoss(size_average=True)
    
    cnn.to(device)
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr,
                                    weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        cnn.train()
        training_distance_list = []
        test_distance_list = []
        training_loss_list = []
        test_loss_list = []
        for i,data in enumerate(train_loader,0):
            imgs,embeddings,_,words = data

            imgs = imgs.to(device)
            embeddings = embeddings.to(device)

            optimizer.zero_grad()
            outputs = cnn(imgs)

            loss = criterion(outputs, embeddings) / batch_size
            loss.backward()
            optimizer.step()

            #training_distance = evaluate_cnn(dream_reader,outputs,words)
            word_distance_array = find_string_distances(outputs.cpu().detach().numpy(),words,pooling_levels,word_length)
            edit_distance_error_avg = float(np.sum(word_distance_array)) / (batch_size*word_length)
            training_loss_list.append(loss.item())
            training_distance_list.append(edit_distance_error_avg)

        print("Epoch : {}, Training loss: {}, Training avg string distance : {}".format(epoch,sum(training_loss_list)/len(training_loss_list),sum(training_distance_list)/len(training_distance_list)))
        #print("End of training epoch :",epoch)
        
        if test_data_path is not None:
            if epoch % 10 == 9:
                cnn.eval()
                for i,data in enumerate(test_loader,0):
                    imgs,embeddings,_,words = data

                    imgs = imgs.to(device)
                    embeddings = embeddings.to(device)

                    outputs = cnn(imgs)
                    test_loss = criterion(outputs, embeddings) / batch_size
                    #test_distance = evaluate_cnn(dream_reader,outputs,words)
                    word_distance_array = find_string_distances(outputs.cpu().detach().numpy(),words,pooling_levels,word_length)
                    edit_distance_error_avg = float(np.sum(word_distance_array)) / (batch_size*word_length)
                    test_loss_list.append(test_loss.item())
                    test_distance_list.append(edit_distance_error_avg)

                print("Epoch : {}, Validation loss: {}, Test avg string distance: {}".format(epoch,sum(test_loss_list)/len(test_loss_list),sum(test_distance_list)/len(test_distance_list)))
        
