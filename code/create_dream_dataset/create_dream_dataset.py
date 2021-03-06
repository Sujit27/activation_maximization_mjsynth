import sys
sys.path.append("../../library")
sys.path.append("../../")
import time
import argparse
import os
import glob
from pathlib import Path
import shutil
import json
#from sklearn.model_selection import train_test_split

import torchvision
from dict_network.dict_net import *
from create_dream import *
from helper_functions import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = None, dest='num_labels',help='Number of labels in the trained dictnet')
parser.add_argument('-b',type=int,default = 512, dest='batch_size',help='batch size for dreaming')
parser.add_argument('-r',type=float,default = 0.1, dest='test_ratio',help='ratio of test data size to be created when train test is split')
parser.add_argument('-m',type=str,default = None, dest='model_name',required=True, help='Trained model for dreaming')
parser.add_argument('-d',type=str,default = None, dest='dict_file',required=True, help='json file with label number to word mapping for the trained dictnet')
parser.add_argument('-o',type=str,default = "out", dest='output_path',help='dream output location')
parser.add_argument('-t',type=bool,default = False, dest='only_test',help='Bool variable to show if data set is being created only for test')

parser.add_argument('-cc',type=int,default = 16, dest='conv_capacity',help='Capacity parameter of convolution layer')
parser.add_argument('-fc',type=int,default = 128, dest='full_capacity',help='Capacity parameter of fully connected layer')
#parser.add_argument('-ne',type=int,default = 10, dest='num_epochs',help='Number of epochs to train')

cmd_args = parser.parse_args()


def main():
    num_labels = cmd_args.num_labels
    batch_size = cmd_args.batch_size
    model_name = cmd_args.model_name
    dict_file = cmd_args.dict_file
    output_path = cmd_args.output_path
    test_ratio = cmd_args.test_ratio
    only_test = cmd_args.only_test
    
    # create paths for output dataset
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    output_path_train = os.path.join(output_path,'train')
    os.makedirs(output_path_train)
    output_path_test = os.path.join(output_path,'test')
    os.makedirs(output_path_test)

    # load the network
    net = DictNet(num_labels,cmd_args.conv_capacity,cmd_args.full_capacity)
    net.load_state_dict(torch.load(model_name)['state_dict'])

    # separate training and test labels
    train_dataset_labels = np.random.choice(num_labels,size=int((1-test_ratio)*num_labels),replace=False)
    test_dataset_labels = list(set(range(num_labels))-set(train_dataset_labels))

    print("Number of labels for training: ", len(train_dataset_labels))
    print("Number of labels for testing: ", len(test_dataset_labels))

    with open(dict_file) as json_file:
        label_dict = json.load(json_file)

    start = time.time()
    total_training_dreams_num = 50000
    num_samples_per_label_training = int(total_training_dreams_num/float(len(train_dataset_labels)))
    if only_test:
        num_samples_per_label_training = 1
    num_samples_per_label_test = 1
    # create training dream dataset
    for i in range(num_samples_per_label_training):
        chunk_size = min(batch_size,len(train_dataset_labels))
        train_dataset_labels_chunked = [train_dataset_labels[i:i + chunk_size] for i in range(0, len(train_dataset_labels), chunk_size)] # create batch size chunks of dataset labels

        for labels in train_dataset_labels_chunked:
            random_seed = np.random.randint(99999999)
            output = dream(net,labels,(32,128,1),(0.47,),(0.14,),random_seed=random_seed)
            save_images(output,labels,label_dict,output_path_train,random_seed)


        print("{} Train dream images per label generated".format(i+1)) 

    # create test dream dataset
    for i in range(num_samples_per_label_test):
        chunk_size = min(batch_size,len(test_dataset_labels))
        test_dataset_labels_chunked = [test_dataset_labels[i:i + chunk_size] for i in range(0, len(test_dataset_labels), chunk_size)] # create batch size chunks of dataset labels

        for labels in test_dataset_labels_chunked:
            random_seed = np.random.randint(99999999)
            output = dream(net,labels,(32,128,1),(0.47,),(0.14,),random_seed=random_seed)
            save_images(output,labels,label_dict,output_path_test,random_seed)


        print("{} Test dream images per label generated".format(i+1)) 


        end = time.time()
    print("Time taken for dreaming and saving images : {} seconds".format((end-start)))
#
    make_dataset_ready_for_PhocNet(output_path_train)
    make_dataset_ready_for_PhocNet(output_path_test)
    
if __name__ == "__main__":
    main()
    
