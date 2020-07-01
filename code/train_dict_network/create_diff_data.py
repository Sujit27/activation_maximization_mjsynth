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
parser.add_argument('-j',type=str,default = None, dest='json_file',required=True,help='Json file from a dict net training. The values of the dictionary in json file are the words that the dict net was trained on. These words will be completely excluded while creating the new data subset')
parser.add_argument('-n',type=int,default = None, dest='num_labels',required=True,help='Number of unique words to be created in the data subset')

args = parser.parse_args()

def create_diff_data_subset(root_dir,num_labels,json_file):
    '''Given a root_dir, number of labels required and a json file whose labels should be excluded
    while creating this new sub dataset, creates a directory data_subset 
    '''
    with open(json_file) as f:
        dict_prev = json.load(f)
    
    word_list_prev = [word for word in dict_prev.values()] # list of previously used labels

    annotation_file = os.path.join(root_dir,'raw','annotation_train.txt')
    with open(annotation_file) as f:
        all_filenames = [line.rstrip() for line in f]
        
    all_words = [((filename.split("_"))[1]).lower() for filename in all_filenames]

    diff_words = list(set(all_words) - set(word_list_prev))
    word_list_new = random.sample(diff_words,num_labels)
    
    # select filnenames that have words from the new list in them
    selected_filenames = [filename for filename in all_filenames if (filename.split("_"))[1].lower() in word_list_new]

    new_dir = os.path.join(root_dir,"data_subset")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(os.path.join(new_dir,'raw'))

    for filename in selected_filenames:
        shutil.copy(os.path.join(root_dir,'raw',filename),os.path.join(new_dir,'raw'))

    with open(os.path.join(new_dir,'raw','annotation_train.txt'),'w') as f:
        for filename in selected_filenames:
            f.write("%s\n" % filename)

def main():
    create_diff_data_subset(args.root_dir,args.num_labels,args.json_file)

if __name__ == "__main__":
    main()
