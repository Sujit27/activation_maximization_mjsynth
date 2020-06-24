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
#from helper_functions import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = None, dest='num_labels',help='Number of labels in the trained dictnet')
parser.add_argument('-b',type=int,default = 512, dest='batch_size',help='batch size for dreaming')
parser.add_argument('-r',type=float,default = 0.1, dest='test_ratio',help='ratio of test data size to be created when train test is split')
parser.add_argument('-s1',type=int,default = 5, dest='num_samples_per_label_training',help='number of samples per label to be generated for training dataset')
parser.add_argument('-s2',type=int,default = 1, dest='num_samples_per_label_test',help='number of samples per label to be generated for test dataset')
parser.add_argument('-m',type=str,default = None, dest='model_name',required=True, help='Trained model for dreaming')
parser.add_argument('-d',type=str,default = None, dest='dict_file',required=True, help='json file with label number to word mapping for the trained dictnet')
parser.add_argument('-o',type=str,default = "out", dest='output_path',help='dream output location')


cmd_args = parser.parse_args()

def create_annotation_txt(files_path):
    '''
    creates a txt file listing the names of all the jpg files
    in the given directory
    '''
    file_list = glob.glob(os.path.join(files_path,"*.jpg"))
    output_file_name = os.path.join(files_path,"annotation_train.txt")

    print("Creating annotation txt file")

    with open(output_file_name,'w') as f:
        for item in file_list:
            f.write("%s\n" % os.path.basename(item))
            
   
def make_dataset_ready_for_PhocNet(data_path):
    '''creates annotation txt file from the name of all image files in the data path provided and 
    creates a new sub directory 'raw' and moves data into it (this is required for phocNet training)
    '''
    create_annotation_txt(data_path)
    raw_path = os.path.join(data_path,'raw')
    Path(raw_path).mkdir(parents=True,exist_ok=True)
    files = os.listdir(data_path)
    for f in files:
        shutil.move(os.path.join(data_path,f),raw_path)

def save_images(tensor,labels,label_dict,output_path,random_seed):
    '''saves a batch of images (BxCxHxW) as B individual image files in the output_path'''
    words = [label_dict[str(label)] for label in labels]
    for j in range(tensor.shape[0]):
        img = tensor[j]
        file_name = str(labels[j]) + "_" + words[j] + "_" + str(random_seed) + ".jpg"
        torchvision.utils.save_image(img,os.path.join(output_path,file_name))


def main():
    num_labels = cmd_args.num_labels
    batch_size = cmd_args.batch_size
    model_name = cmd_args.model_name
    dict_file = cmd_args.dict_file
    output_path = cmd_args.output_path
    test_ratio = cmd_args.test_ratio
    num_samples_per_label_training = cmd_args.num_samples_per_label_training
    num_samples_per_label_test = cmd_args.num_samples_per_label_test
    
    # create paths for output dataset
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    output_path_train = os.path.join(output_path,'train')
    os.makedirs(output_path_train)
    output_path_test = os.path.join(output_path,'test')
    os.makedirs(output_path_test)

    # load the network
    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name)['state_dict'])

    # separate training and test labels
    train_dataset_labels = np.random.choice(num_labels,size=int((1-test_ratio)*num_labels),replace=False)
    test_dataset_labels = list(set(range(num_labels))-set(train_dataset_labels))

    print("Number of labels for training: ", len(train_dataset_labels))
    print("Number of labels for testing: ", len(test_dataset_labels))

    with open(dict_file) as json_file:
        label_dict = json.load(json_file)

    start = time.time()
    
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
    
