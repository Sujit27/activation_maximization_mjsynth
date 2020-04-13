#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../library")
sys.path.append("../library/phoc_net/")
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from phoc_net import *
from phoc_dataset import *
from phoc_embedding import *
from cosine_loss import *
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d',type=str,default = "../sample_dreams_test/", dest='test_data_root',help='location of test dataset')
parser.add_argument('-m',type=str,default = "../models2/PhocNet_8_1000.pt", dest='phoc_net_file',help='trained phocNet model file location')
parser.add_argument('-l',type=str,default = "../lexicon.txt", dest='lex_file',help='location of the full lexicon txt file')

args = parser.parse_args()

def predict_index(output,voc_embedding,voc_dict):
    ''' tests whether the phocNet predicts the correct words for a dataset and outputs the accuracy score'''
    output = np.transpose(output.cpu().detach().numpy())
    similarity = np.dot(voc_embedding,output)
    
    indices_pred = np.argmax(similarity,0)  
    indices_pred = np.array(indices_pred).flatten()

    predicted_words = [voc_dict[index] for index in indices_pred]
    
    return indices_pred,predicted_words

def word_to_index(dictionary,words):
    indices = [dictionary[word] for word in words]
    return np.asarray(indices)

def main():
    
    test_data_root = args.test_data_root
    phoc_net_file = args.phoc_net_file
    lex_file = args.lex_file

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # create test data loader
    test_data_set = PhocDataset(test_data_root)
    batch_size = 8
    test_loader = DataLoader(test_data_set,batch_size=batch_size,num_workers=8)
    
    # create the network
    phoc_net = PHOCNet(test_data_set[0][1].shape[0],pooling_levels=4)
    phoc_net.load_state_dict(torch.load(phoc_net_file))
    phoc_net.eval()
    phoc_net.to(device)
    
    # create vocab embeddings
    with open(lex_file) as f:
        voc = f.readlines()
    voc = [word[:-1] for word in voc]
    voc_dict = {index:word for (index, word) in enumerate(voc)}
    voc_inv_dict = {word:index for (index, word) in enumerate(voc)}
    unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    phoc_unigram_levels=(1, 2, 4, 8)
    voc_embedding = build_phoc_descriptor(voc,unigrams,phoc_unigram_levels)
    
    for i,data in enumerate(test_loader,0):
        imgs,embeddings,class_ids,words = data
        imgs = imgs.to(device)
        
        output = F.sigmoid(phoc_net(imgs))
        predicted_indices,predicted_words = predict_index(output,voc_embedding,voc_dict)
        
        actual_indices = word_to_index(voc_inv_dict,words)
        
        acc_score = accuracy_score(actual_indices,predicted_indices)
        print("{} : {}".format(i,acc_score))
        #print(words)
        #print(predicted_words)
        #break
        


if __name__ == "__main__":
    main()



