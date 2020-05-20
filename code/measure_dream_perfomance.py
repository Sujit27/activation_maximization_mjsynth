import sys
sys.path.append('../library')
import time

from create_dream import *

'''Creates one batch of dreams with a trained dictnet by varying the batch size and 
    displays the time taken for each batch size'''

def batch_dream(num_labels):
    labels = [i for i in range(num_labels)]
    nItr = 100

    start = time.time()
    output_batch = dream(network,labels,(32,128,1),(0.47,),(0.14,),nItr)
    end = time.time()
    
    print("{}, {}".format(output_batch.shape[0],end-start))

if __name__ == "__main__":
    network = DictNet(1000)
    network.load_state_dict(torch.load("../code/train_dict_network/out3/net_1000_0.001_200_0.0.pth"))
    num_labels_list = [1,2,4,8,16,32,64,128,256,512]
    print("batch_size, dream_time(sec)")
    for num_labels in num_labels_list:
        batch_dream(num_labels)
