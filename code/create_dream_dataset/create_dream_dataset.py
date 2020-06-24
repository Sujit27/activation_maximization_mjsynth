import sys
sys.path.append("../../library")
sys.path.append("../../")
import time
import argparse
import os
import glob
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

import torchvision
from dict_network.dict_net import *
from create_dream import *
from helper_functions import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = 512, dest='num_samples',help='Number of dream samples to be generated')
parser.add_argument('-b',type=int,default = 512, dest='batch_size',help='batch size for dreaming')
parser.add_argument('-r',type=float,default = 0.3, dest='test_ratio',help='ratio of test data size to be created when train test is split')
parser.add_argument('-m',type=str,default = None, dest='model_name',required=True, help='Trained model for dreaming')
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
            
def split_dream_dataset(data_path,test_ratio):
    '''Given location which contains dream images in the format #_word_#.jpg and a ratio, the function
    finds the number of unique word labels and splits them into training labels and test labels. It then
    splits all the dream image file names present in the directory into training_files and test_files, creates 
    train and test directories and copies the corresponding images to each directory
    '''
    file_list = glob.glob(os.path.join(data_path,"*.jpg"))
    labels = [os.path.basename(file).split("_")[1] for file in file_list]
    unique_labels = list(set(labels))
    label_train ,label_test = train_test_split(unique_labels,test_size=test_ratio)
    
    train_file_list = []
    test_file_list = []
    
    for file in file_list:
        for label in label_train:
            if label in file:
                train_file_list.append(file)
                
    for file in file_list:
        for label in label_test:
            if label in file:
                test_file_list.append(file)
    
    dir_train = 'train'
    if os.path.exists(dir_train):
        shutil.rmtree(dir_train)
    os.makedirs(dir_train)
    
    dir_test = 'test'
    if os.path.exists(dir_test):
        shutil.rmtree(dir_test)
    os.makedirs(dir_test)
    
    for full_file_name in train_file_list:
        shutil.copy(full_file_name,dir_train)
        
    for full_file_name in test_file_list:
        shutil.copy(full_file_name,dir_test)
        
    return label_train ,label_test
    
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

def save_images(tensor,labels,output_path,prev_serial_num,random_seed):
    '''saves a batch of images (BxCxHxW) as B individual image files in the output_path'''
    words = label_to_word(labels)
    for j in range(tensor.shape[0]):
        img = tensor[j]
        file_name = str(prev_serial_num + j) + "_" + words[j] + "_" + str(random_seed) + ".jpg"
        torchvision.utils.save_image(img,os.path.join(output_path,file_name))


def main():
    num_samples = cmd_args.num_samples
    batch_size = cmd_args.batch_size
    model_name = cmd_args.model_name
    output_path = cmd_args.output_path
    test_ratio = cmd_args.test_ratio

    model_basename = os.path.basename(model_name)
    num_labels = int(model_basename.split('_')[1])

    #Path(output_path).mkdir(parents=True,exist_ok=True)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name))

    start = time.time()

    for i in range(num_samples // batch_size):
        random_seed = np.random.randint(99999999)
        labels = np.random.randint(num_labels,size=batch_size)
        output = dream(net,labels,(32,128,1),(0.47,),(0.14,),random_seed=random_seed)

        save_images(output,labels,output_path,i*batch_size,random_seed)

        end = time.time()
    print("Time taken for dreaming and saving {} images : {} seconds".format((i+1)*batch_size,end-start))

    label_train ,label_test = split_dream_dataset(output_path,test_ratio)
    make_dataset_ready_for_PhocNet('train')
    make_dataset_ready_for_PhocNet('test')
    
if __name__ == "__main__":
    main()
    
