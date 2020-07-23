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
from softmax_filter import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = None, dest='num_labels',help='Number of labels in the trained dictnet')
parser.add_argument('-b',type=int,default = 512, dest='batch_size',help='batch size for dreaming')
parser.add_argument('-m',type=str,default = None, dest='model_name',required=True, help='Trained model for dreaming')
parser.add_argument('-d',type=str,default = None, dest='dict_file',required=True, help='json file with label number to word mapping for the trained dictnet')
parser.add_argument('-c',type=str,default = None, dest='csv_file',required=True, help='csv file that has three columns for prediction on rendered images: label_predicted,softmax probability and word caption of the rendered image')
parser.add_argument('-t',type=float,default = None, dest='threshold_prob',help='threshold limit for filterring labels to be chosen')
parser.add_argument('-o',type=str,default = "out", dest='output_path',help='dream output location')


cmd_args = parser.parse_args()


def main():
    num_labels = cmd_args.num_labels
    batch_size = cmd_args.batch_size
    model_name = cmd_args.model_name
    dict_file = cmd_args.dict_file
    csv_file = cmd_args.csv_file
    threshold_prob = cmd_args.threshold_prob
    output_path = cmd_args.output_path
    
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

    # filter the labels to be used for training and evaluation
    labels_guessed,correct_labels_guessed = get_softmax_correct_labels(csv_file,dict_file,threshold_prob)
    print("Number of labels guessed :{}, number of correct ones : {}".format(len(labels_guessed),len(correct_labels_guessed)))
    #separate training and test labels
    train_dataset_labels = [label for label in labels_guessed.keys()] 
    test_dataset_labels = list(set(range(num_labels))-set(train_dataset_labels))

    print("Number of labels for training: ", len(train_dataset_labels))
    print("Number of labels for testing: ", len(test_dataset_labels))

    with open(dict_file) as json_file:
        label_dict = json.load(json_file)

    start = time.time()
    total_training_dreams_num = 50000
    num_samples_per_label_training = int(total_training_dreams_num/float(len(train_dataset_labels)))
    num_samples_per_label_test = 1
    # create training dream dataset
    for i in range(num_samples_per_label_training):
        chunk_size = min(batch_size,len(train_dataset_labels))
        train_dataset_labels_chunked = [train_dataset_labels[i:i + chunk_size] for i in range(0, len(train_dataset_labels), chunk_size)] # create batch size chunks of dataset labels

        for labels in train_dataset_labels_chunked:
            random_seed = np.random.randint(99999999)
            output = dream(net,labels,(32,128,1),(0.47,),(0.14,),random_seed=random_seed)
            words = [labels_guessed[label] for label in labels]
            for j in range(output.shape[0]):
                img = output[j]
                file_name = str(labels[j]) + "_" + words[j] + "_" + str(random_seed) + ".jpg"
                torchvision.utils.save_image(img,os.path.join(output_path_train,file_name))


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
    
