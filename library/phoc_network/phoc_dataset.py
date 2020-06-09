import sys
sys.path.append('../')
import os

import numpy as np
from skimage import io as img_io
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

from skimage.transform import resize

import dagtasets as dg
from helper_functions import *
from phoc_embedding import build_phoc_descriptor, get_unigrams_from_strings
#from cnn_ws.transformations.image_size import check_size
#from cnn_ws.transformations.homography_augmentation import HomographyAugmentation


class PhocDataset(Dataset):
    '''
    Phoc dataset class for the mjsynth dataset
    '''

    def __init__(self, root_dir,phoc_unigram_levels=(2, 4, 6, 8)):
        '''root_dir : location of dataset
        '''
        # class members
        self.word_list = None
        self.word_string_embeddings = None
        self.query_list = None
        self.label_encoder = None

        transform = dg.mjsynth.mjsynth_gray_scale
        ds = dg.mjsynth.MjSynthWS(root_dir,transform)
        self.ds = ds
        
        annotation_file = os.path.join(root_dir,'raw',"annotation_train.txt")
        with open(annotation_file) as f:
            lines = [line.rstrip() for line in f]
        words = [((line.split("_"))[1]).lower() for line in lines]
#        words = [[key[1]]*len(value) for key,value in words_dict.items()]
#        words = [word for sublist in words for word in sublist]

        # compute a mapping from class string to class id
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(words)

        # extract unigrams from train split
        unigrams = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        
        word_embeddings = build_phoc_descriptor(words=words,phoc_unigrams=unigrams,unigram_levels=phoc_unigram_levels)
        
        self.word_list = words
        self.word_string_embeddings = word_embeddings.astype(np.float32)

    def embedding_size(self):
        return len(self.word_string_embeddings[0])


    def __len__(self):
        return len(self.word_list)

    def __getitem__(self, index):
        word_img, _ = self.ds[index]
        embedding = self.word_string_embeddings[index]
        embedding = torch.from_numpy(embedding)
        class_id = self.label_encoder.transform([self.word_list[index]])
        word = self.ds.filenames[index].split("_")[1]

        return word_img, embedding, class_id, word 

