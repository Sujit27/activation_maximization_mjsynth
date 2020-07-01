import sys

import os
import shutil
import random

from sortedcontainers import SortedSet

import dagtasets as dg

import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


def create_data_subset(root_dir,num_labels):
    '''
    creates a different directory at the root_dir with the name data_subset
    which has a subset of the images at root_dir/raw according to the number of 
    labels(words) specified
    '''
    annotation_file = os.path.join(root_dir,'raw','annotation_train.txt')
    with open(annotation_file) as f:
        all_filenames = [line.rstrip() for line in f]
    words = [((filename.split("_"))[1]).lower() for filename in all_filenames]

    word_list = list(SortedSet(words))
    random.seed(42)
    if len(word_list) < num_labels:
        print("Number of labels requested is greater than number of unique words present at root. Data subset not possible")
        return
    else:
        #word_list = word_list[:num_labels]
        word_list = random.sample(word_list,num_labels)

        
    selected_filenames = [filename for filename in all_filenames if (filename.split("_"))[1].lower() in word_list]

    new_dir = os.path.join(root_dir,"data_subset")
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(os.path.join(new_dir,'raw'))

    for filename in selected_filenames:
        shutil.copy(os.path.join(root_dir,'raw',filename),os.path.join(new_dir,'raw'))

    with open(os.path.join(new_dir,'raw','annotation_train.txt'),'w') as f:
        for filename in selected_filenames:
            f.write("%s\n" % filename)
        


class DictNetDataset(Dataset):
    '''
    Dataset for training DictNet
    '''

    def __init__(self,root_dir,num_labels=None):

        if num_labels is not None:
            create_data_subset(root_dir,num_labels)
            root_dir = os.path.join(root_dir,'data_subset')

        transform = dg.mjsynth.mjsynth_gray_scale # gray scale and resize to 32x128 transform
        self.ds = dg.mjsynth.MjSynthWS(root_dir,transform)

        annotation_file = os.path.join(root_dir,'raw','annotation_train.txt')
        with open(annotation_file) as f:
            lines = [line.rstrip() for line in f]
        self.words = [((line.split("_"))[1]).lower() for line in lines]
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.words)

        self.num_labels = len(list(self.label_encoder.classes_))

        unique_words = list(self.label_encoder.inverse_transform(range(self.num_labels)))
        self.label_dict = dict(zip(range(self.num_labels),unique_words))


    def __len__(self):
        return len(self.words)


    def __getitem__(self,index):
        tensor,_ = self.ds[index]
        word = self.words[index]
        label = self.label_encoder.transform([word])[0]

        return tensor,label
