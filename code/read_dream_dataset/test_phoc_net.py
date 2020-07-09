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
parser.add_argument('-hsg',type=str,default='Histogram',dest='histogram_title',help='Title of histogram plot generated showing distribution of edit distance error on test set')

args = parser.parse_args()

def main():
    # load test datset and trained phoc net model
    test_data_path = os.path.join(args.test_data_path,'test')
    loaded_model = torch.load(args.trained_phoc_net_model)
    
    phoc_pooling_levels = [1,2,3,4,5,6,7,8,9,10]

    # create dataloader and load wieghts into phoc net   
    test_data_set = PhocDataset(test_data_path,phoc_pooling_levels)
    phoc_net = PHOCNet(test_data_set[0][1].shape[0]+10,phoc_pooling_levels,input_channels=1,gpp_type='tpp')
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

        outputs = torch.sigmoid(phoc_net(imgs))
        word_dist_array = find_string_distances(outputs.cpu().detach().numpy()[:,:-10],words,phoc_pooling_levels)
        edit_distance_list.append(word_dist_array)
    
    edit_distance_list = [distance for word_dist_array in edit_distance_list for distance in word_dist_array]

    # plot distribution of edit distances from gorund truth
    hist_freq = plt.hist(edit_distance_list,bins=range(11))
    plt.figure(num=None,figsize=(8,6),dpi=80,facecolor='w',edgecolor='k')
    plt.bar(hist_freq[1][:-1], hist_freq[0]/sum(hist_freq[0])*100)
    plt.title("Edit Distance Distribution on Test Set\n {}".format(args.histogram_title))
    plt.xlim(0,10)
    plt.ylim(0,100)
    plt.xlabel("edit distance from ground truth")
    plt.ylabel("Frequency (percent)")
    plt.savefig("{}.png".format(args.histogram_title))

    #print(hist_array)

if __name__ == "__main__":
    main()
