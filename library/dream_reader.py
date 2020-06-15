import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from phoc_network.phoc_net import *
#from .phoc_network.phoc_dataset import *
from phoc_network.phoc_embedding import *
from phoc_network.cosine_loss import *
#from sklearn.metrics import accuracy_score


def predict_index(output,voc_embedding,voc_dict):
    ''' given output from a phocnet, embedding of all vocab in the lexicon and 
    dict of index to word, returns the indices and words corresponding to
    the output'''
    output = np.transpose(output.cpu().detach().numpy())
    similarity = np.dot(voc_embedding,output)
    
    indices_pred = np.argmax(similarity,0)  
    indices_pred = np.array(indices_pred).flatten()

    predicted_words = [voc_dict[index] for index in indices_pred]
    
    return indices_pred,predicted_words

def word_to_index(inv_voc_dict,words):
    indices = [inv_voc_dict[word] for word in words]
    return np.asarray(indices)


class DreamReader():
    '''Given a lexicon that covers all possible words, a phocNet
    a dream reader object should read a batch of images and output
    the words in those images
    '''
    def __init__(self,lex_list,phoc_unigram_levels=(2, 4, 6, 8),phoc_net=None,):
        self.voc = lex_list
        self.phoc_net = phoc_net
        self.phoc_unigram_levels = phoc_unigram_levels
        self.unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]

        self.voc_dict = {index:word for (index, word) in enumerate(self.voc)}
        self.voc_inv_dict = {word:index for (index, word) in enumerate(self.voc)}
        self.voc_embedding = build_phoc_descriptor(self.voc,self.unigrams,self.phoc_unigram_levels)


    def read_from_network_output(self,output):
        return predict_index(output,self.voc_embedding,self.voc_dict)

    def read_from_images(self,image_tensor,network=None):
        if network is None:
            network = self.phoc_net
        if next(network.parameters()).is_cuda:
            image_tensor = image_tensor.cuda()

        output = F.sigmoid(network(image_tensor))
        
        return read_from_network_output(output)

    def convert_words_to_indices(self,words):
        
        return word_to_index(self.voc_inv_dict,words)
