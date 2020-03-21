#!/usr/bin/env python

import sys
sys.path.append("../library/")
from dict_net import *
from deep_dream import *
from helper_functions import *
from discrim_net import *
from glue_layers import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m',type=str,default = "../models/net_1000_0.001_200_0.0.pth", dest='trained_model',help='Full path of existing dreamer network')
parser.add_argument('-lr',type=float,default = 0.001, dest='lr',help='Learning rate')
parser.add_argument('-bs',type=int,default = 32, dest='batch_size',help='Batch size')
parser.add_argument('-d',type=str,default = "/var/tmp/on63ilaw/mjsynth/", dest='data_root',help='input real data location')
parser.add_argument('-gl',type=bool,default = False, dest='use_glue_layer',help='whether to use a simple glue layer or not')

cmd_args = parser.parse_args()

def main():
    trained_model = cmd_args.trained_model
    lr = cmd_args.lr
    batch_size = cmd_args.batch_size
    data_root = cmd_args.data_root
    use_glue_layer = cmd_args.use_glue_layer
    
    filename = os.path.basename(trained_model)
    num_labels = int(filename.split('_')[1])

    # set dreamer network
    net = DictNet(num_labels)
    net.load_state_dict(torch.load(trained_model))

    #set glue layer
    if use_glue_layer :
        glue_net = BasicGlueLayer()
    else:
        glue_net = None

    #Initialize dreamer
    dreamer = DeepDreamGAN(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True,glue_layer=glue_net,discrim_net_use_gaussian_filter=False)

    #set dataset
    transform = dg.mjsynth.mjsynth_gray_scale
    ds, labels_and_indices_dict, labels_dict, labels_inv_dict = create_dicts(data_root,transform)
    ds = extract_dataset(ds,labels_and_indices_dict,labels_dict,num_labels,prev_num_labels=0)


    #train dreamer
    dreamer.train_model(ds,lr=lr,batch_size=batch_size,stop_training_batch_num=30)

if __name__ == "__main__":
    main()
