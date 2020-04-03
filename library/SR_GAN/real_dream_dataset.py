import random
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

        return real_image,dream_image

