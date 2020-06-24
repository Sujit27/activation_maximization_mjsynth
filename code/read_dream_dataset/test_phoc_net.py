# coding: utf-8
import sys
sys.path.append("../../library/phoc_network/")
import os
from phoc_net import *
from predict_word_from_embd import *
from phoc_dataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d',type=str,default=None,dest='test_data_path',required=True,help='Loaction of test dream dataset. Location should have a dir called test in it')
parser.add_argument('-m',type=str,default=None,dest='trained_phoc_net_model',required=True,help='Loaction of trained phoc net model : .pth.tar file')
parser.add_argument('-wl',type=int,default = None, dest='word_length',required=True,help='Length of the word whose image is to be read.Should be between [1,10]')

args = parser.parse_args()

def main():
    # load test datset and trained phoc net model
    test_data_path = os.path.join(args.test_data_path,'test')
    loaded_model = torch.load(args.trained_phoc_net_model)
    
    # set pooling levels according to word length
    if args.word_length % 2 == 0: # word_length is even
        phoc_pooling_levels = [2,4,6,8,10]
    else:
        phoc_pooling_levels = [1,3,5,7,9]
   
    # create dataloader and load wieghts into phoc net   
    test_data_set = PhocDataset(test_data_path,phoc_pooling_levels)
    phoc_net = PHOCNet(test_data_set[0][1].shape[0],phoc_pooling_levels,input_channels=1,gpp_type='tpp')
    phoc_net.load_state_dict(loaded_model['state_dict'])
    batch_size = min(32,len(test_data_set)) # batch size 32 or length of dataset for testing
    test_loader = DataLoader(test_data_set,batch_size = batch_size)

    # evaluate phoc net on test data
    phoc_net.eval()
    phoc_net.to(torch.device('cuda'))
    edit_distance_list = []
    for i,data in enumerate(test_loader,0):
        imgs,embeddings,_,words = data
        imgs = imgs.to(torch.device('cuda'))
        embeddings = embeddings.to(torch.device('cuda'))
        outputs = phoc_net(imgs)
        word_dist_array = find_string_distances(outputs.cpu().detach().numpy(),words,phoc_pooling_levels,args.word_length)
        edit_distance_list.append(word_dist_array)
    
    edit_distance_list = [distance for word_dist_array in edit_distance_list for distance in word_dist_array]
    hist_array = plt.hist(edit_distance_list)

    print(hist_array)

if __name__ == "__main__":
    main()
