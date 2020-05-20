import sys
sys.path.append('../library')
import time

from create_dream import *
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = 4, dest='num_labels',help='Number of labels in the batch')

cmd_args = parser.parse_args()

def main():
    network = DictNet(1000)
    network.load_state_dict(torch.load("../code/train_dict_network/out3/net_1000_0.001_200_0.0.pth"))
    labels = [i for i in range(cmd_args.num_labels)]
    nItr = 100

    start = time.time()
    output_batch = dream(network,labels,(32,128,1),(0.47,),(0.14,),100)
    end = time.time()
    
    print("Time to dream a batch of {} mjsynth images: {}".format(output_batch.shape[0],end-start))

if __name__ == "__main__":
    main()
