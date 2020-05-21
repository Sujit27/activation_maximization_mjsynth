import sys
sys.path.append("../../library")
sys.path.append("../../")
import time
import argparse
import os
import glob
from pathlib import Path
import shutil

import torchvision
from dict_network.dict_net import *
from create_dream import *
from helper_functions import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = 512, dest='num_samples',help='Number of dream samples to be generated')
parser.add_argument('-b',type=int,default = 512, dest='batch_size',help='batch size for dreaming')
parser.add_argument('-m',type=str,default = None, dest='model_name',help='Trained model for dreaming')
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

def save_images(tensor,labels,output_path,prev_serial_num,random_seed):
    '''saves a batch of images (BxCxHxW) as B individual image files in the output_path'''
    words = label_to_word(labels)
    for j in range(tensor.shape[0]):
        img = tensor[j]
        file_name = str(prev_serial_num + j) + "_" + words[j] + "_" + str(random_seed) + ".png"
        torchvision.utils.save_image(img,os.path.join(output_path,file_name))


def main():
    num_samples = cmd_args.num_samples
    batch_size = cmd_args.batch_size
    model_name = cmd_args.model_name
    output_path = cmd_args.output_path

    model_basename = os.path.basename(model_name)
    num_labels = int(model_basename.split('_')[1])

    Path(output_path).mkdir(parents=True,exist_ok=True)

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

    create_annotation_txt(output_path)
    raw_path = os.path.join(output_path,'raw')
    Path(raw_path).mkdir(parents=True,exist_ok=True)
    files = os.listdir(output_path)
    for f in files:
        shutil.move(os.path.join(output_path,f),raw_path)

    
if __name__ == "__main__":
    main()
    
