import sys
sys.path.append("../../library")
sys.path.append("../../library/dict_network")

import dagtasets as dg
from train_dictnet import *
from torch.utils.data  import SubsetRandomSampler
from torch import optim
import csv
import os
import glob
from statistics import mean
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = None, dest='num_labels',help='Store number of labels')
parser.add_argument('-m',type=str,default = None, dest='prev_trained_checkpoint',help='Full path of existing model')
parser.add_argument('-o',type=str,default = "out", dest='output_path',help='Output model location')
parser.add_argument('-d',type=str,default = None, dest='data_root',help='input data location')
parser.add_argument('-lr',type=float,default = 0.001, dest='lr',help='Learning rate')
#parser.add_argument('-bs',type=int,default = 64, dest='batch_size',help='batch size for training')
parser.add_argument('-ne',type=int,default = 30, dest='num_epochs',help='Number of epochs to train')


cmd_args = parser.parse_args()



   
def main():
    print(cmd_args)
    # Can either train a model from start given the number of labels 
    # Or can grow an existing trained model ( see library/dict_network for more details ) by increasing the number of labels to a specified number 
    num_labels = cmd_args.num_labels
    prev_trained_checkpoint = cmd_args.prev_trained_checkpoint
    output_path = cmd_args.output_path
    data_root = cmd_args.data_root
    lr = cmd_args.lr
    num_epochs = cmd_args.num_epochs

    os.nice(20)

    weight_decay = 0.00
    transform = dg.mjsynth.mjsynth_gray_scale
    
    if num_labels  is not None:
        batch_size = int(num_labels/5) 
        batch_size = min(batch_size,256)
    else:
        batch_size = 256

    # create output directory for saving model if does not exist already
    Path(output_path).mkdir(parents=True,exist_ok=True)

    # train
    train_model(output_path,data_root,transform,prev_trained_checkpoint=prev_trained_checkpoint,num_labels=num_labels,lr = lr,batch_size=batch_size,weight_decay=weight_decay,num_epochs=num_epochs)



if __name__ == "__main__":
    main()
