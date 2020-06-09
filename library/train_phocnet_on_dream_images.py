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
from dream_reader import *


def evaluate_cnn(dream_reader,output,words):
    indices_pred,words_pred = dream_reader.read_from_network_output(output)
    indices_target = dream_reader.convert_words_to_indices(words)

    acc_score = accuracy_score(indices_target,indices_pred)

    return acc_score
    
def train_phocNet_on_dream_dataset(training_data_path,test_data_path,pooling_levels=[2,4,6,8],
                                   num_epochs=100,lr=0.0001,batch_size=64,weight_decay=0.000,
                                   lex_txt_file = "../lexicon.txt",device=torch.device('gpu')):
    
    train_data_set = PhocDataset(training_data_path)
    train_loader = DataLoader(train_data_set,batch_size=batch_size,shuffle=True,num_workers=0)
    
    test_data_set = PhocDataset(test_data_path)
    test_loader = DataLoader(test_data_set,batch_size=batch_size,shuffle=True,num_workers=0)
    
    with open(lex_txt_file) as f:
        lex_list = f.readlines()
    lex_list = [word[:-1] for word in lex_list] 
    dream_reader = DreamReader(lex_list)
    
    cnn = PHOCNet(n_out=train_data_set[0][1].shape[0],input_channels=1,gpp_type='tpp',pooling_levels=pooling_levels)
    cnn.init_weights()
    criterion = nn.BCEWithLogitsLoss(size_average=True)
    
    cnn.to(device)
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr,
                                    weight_decay=weight_decay)
    
    for epoch in range(num_epochs):
        cnn.train()
        training_acc_list = []
        test_acc_list = []
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

            training_acc = evaluate_cnn(dream_reader,outputs,words)
            training_loss_list.append(loss.item())
            training_acc_list.append(training_acc)

        print("Epoch : {}, Training loss: {}, Training accuracy: {}".format(epoch,sum(training_loss_list)/len(training_loss_list),sum(training_acc_list)/len(training_acc_list)))
        #print("End of training epoch :",epoch)
        
        if epoch % 10 == 9:
            cnn.eval()
            for i,data in enumerate(test_loader,0):
                imgs,embeddings,_,words = data

                imgs = imgs.to(device)
                embeddings = embeddings.to(device)

                outputs = cnn(imgs)
                test_loss = criterion(outputs, embeddings) / batch_size
                test_acc = evaluate_cnn(dream_reader,outputs,words)
                test_loss_list.append(test_loss.item())
                test_acc_list.append(test_acc)

            print("Epoch : {}, Validation loss: {}, Validation accuracy: {}".format(epoch,sum(test_loss_list)/len(test_loss_list),sum(test_acc_list)/len(test_acc_list)))

