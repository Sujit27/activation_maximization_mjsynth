import sys
sys.path.append("../../library")
sys.path.append("../../library/phoc_network")
import argparse

from train_phocnet_on_dream_images import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-wl',type=int,default = None, dest='word_length',required=True,help='Length of the word whose image is to be read.Should be between [1,10]') 
parser.add_argument('-bs',type=int,default = 64, dest='batch_size',help='batch size for training phocnet')
parser.add_argument('-ne',type=int,default = 50, dest='num_epochs',help='number of epochs for training phocnet')
parser.add_argument('-lr',type=float,default = 0.0001, dest='learning_rate',help='learning rate for training phocnet')

args = parser.parse_args()

def main():
    training_data_path = "../create_dream_dataset/train"
    test_data_path = "../create_dream_dataset/test"

    if args.word_length % 2 == 0: # word_length is even
        phoc_pooling_levels = [2,4,6,8,10]
    else:
        phoc_pooling_levels = [1,3,5,7,9]

    train_phocNet_on_dream_dataset(phoc_pooling_levels,args.word_length,training_data_path,test_data_path,args.num_epochs,args.learning_rate,args.batch_size)

if __name__ == "__main__":
    main()
