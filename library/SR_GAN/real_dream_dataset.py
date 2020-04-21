import os
import random
from sklearn.preprocessing import LabelEncoder
import torch
import dagtasets as dg
from torch.utils.data import Dataset

class RealDreamDataset(Dataset):
    '''
    creates a dataset by combining real and dream images available in root dirs
    '''
    def __init__(self,real_dataroot="/var/tmp/on63ilaw/mjsynth",dream_dataroot="/var/tmp/on63ilaw/mjsynth/sample_dreams_dataset",transform=dg.mjsynth.mjsynth_gray_scale):
        self.real_dataroot = real_dataroot
        self.dream_dataroot = dream_dataroot
        self.real_dataset = dg.mjsynth.MjSynthWS(self.real_dataroot,transform)
        self.dream_dataset = dg.mjsynth.MjSynthWS(self.dream_dataroot,transform)
        self.transform = transform

        self.real_dataset = self.truncate_real_dataset(self.real_dataset,len(self.dream_dataset))
        
        annotation_file = os.path.join(dream_dataroot,'raw',"annotation_train.txt")
        with open(annotation_file) as f:
            lines = [line.rstrip() for line in f]
        words = [((line.split("_"))[1]).lower() for line in lines]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(words)


    def truncate_real_dataset(self,dataset,reduced_size):
        indices_list = [i for i in range(len(dataset))]
        random_indices = random.sample(indices_list,reduced_size)
        dataset = torch.utils.data.Subset(dataset,random_indices)

        return dataset

    def __len__(self):
        return len(self.dream_dataset)

    def __getitem__(self,index):
        real_image,_ = self.real_dataset[index]
        dream_image,_ = self.dream_dataset[index]

        word = str.lower(self.dream_dataset.filenames[index].split("_")[1])
        target_label = self.label_encoder.transform([word])

        #return real_image,dream_image,target_label[0]
        return real_image,dream_image,word

