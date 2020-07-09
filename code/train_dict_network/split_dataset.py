# coding: utf-8

import sys
import os
import shutil
import json
from sortedcontainers import SortedSet
import random
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d',type=str,default = None, dest='root_dir',required=True,help='Location of mjsynth data')
parser.add_argument('-n',type=int,default = None, dest='num_labels',required=True,help='Number of unique words to be created in the data subset')
parser.add_argument('-l',type=str,default = 'taboo_words.txt', dest='txt_file',help='Text file containing list of words that need to be included in the sequestered dataset')
args = parser.parse_args()

def create_dataset(filenames,root_dir,dataset_name):
    new_dir = os.path.join(root_dir,dataset_name)
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(os.path.join(new_dir,'raw'))

    for filename in filenames:
        shutil.copy(os.path.join(root_dir,'raw',filename),os.path.join(new_dir,'raw'))

    with open(os.path.join(new_dir,'raw','annotation_train.txt'),'w') as f:
        for filename in filenames:
            f.write("%s\n" % filename)


def split_data(root_dir,num_labels,txt_file):
    '''Given a root_dir, number of labels splits the dataset into two datasets Known and Sequestered which has mutually exclusive transcripts. If a txt file containing a list of words is provided,these words in the root are included in the sequestered dataset 
    '''
    annotation_file = os.path.join(root_dir,'raw','annotation_train.txt')
    with open(annotation_file) as f:
        all_filenames = [line.rstrip() for line in f]
        
    all_words = [((filename.split("_"))[1]).lower() for filename in all_filenames]

    unique_words_dataset = set(all_words)

    taboo_words_num = 0
    # add taboo words in sequestered list if txt file provided
    if txt_file is not None:
        with open(txt_file) as f:
            taboo_words = [line.rstrip() for line in f]
        unique_words_taboo = unique_words_dataset.intersection(set(taboo_words))
        unique_words_dataset = unique_words_dataset - unique_words_taboo
        taboo_words_num = len(unique_words_taboo)

    random.seed(0)
    unique_words_sequestered = random.sample(list(unique_words_dataset),num_labels-taboo_words_num)
    if taboo_words_num !=0 :
        unique_words_sequestered = list(set(unique_words_sequestered).union(unique_words_taboo))

    random.seed(0)
    unique_words_known = random.sample(list(unique_words_dataset - set(unique_words_sequestered)),num_labels)
    
    # select filnenames that have words from the new list in them
    filenames_known = [filename for filename in all_filenames if (filename.split("_"))[1].lower() in unique_words_known]
    filenames_sequestered = [filename for filename in all_filenames if (filename.split("_"))[1].lower() in unique_words_sequestered]
    create_dataset(filenames_known,root_dir,'known')
    create_dataset(filenames_sequestered,root_dir,'sequestered')

def main():
    split_data(args.root_dir,args.num_labels,args.txt_file)

if __name__ == "__main__":
    main()
