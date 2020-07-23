import sys
sys.path.append("../../library/phoc_network/")
import os
import json
from phoc_net import *
from predict_word_from_embd import *
from phoc_dataset import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def save_accurate_seq_dict(test_data_path,phoc_network_path):
    
    phoc_pooling_levels = [1,2,3,4,5,6,7,8,9,10]

    # create dataloader and load wieghts into phoc net   
    test_data_set = PhocDataset(test_data_path,phoc_pooling_levels)
    phoc_net = PHOCNet(test_data_set[0][1].shape[0]+10,phoc_pooling_levels,input_channels=1,gpp_type='tpp')
    phoc_net.load_state_dict(torch.load(phoc_network_path)['state_dict'])
    batch_size = 1 # batch size 32 or length of dataset for testing
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

    accurate_test_words = {i:test_data_set[i][3] for i in range(len(edit_distance_list)) if edit_distance_list[i] == 0}
    with open('correct_in_sequestered.json','w') as f:
        json.dump(accurate_test_words,f)
    
def main():
    test_data_path = "../create_dream_dataset/out_dream_dataset_all_5000_sequestered/test"
    phoc_network_path = "out_phocnet_all_5000_known/model_best.pth.tar"
    save_accurate_seq_dict(test_data_path,phoc_network_path)

if __name__ == "__main__":
    main()
