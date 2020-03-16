#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
sys.path.append("../library")
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from dict_net import *
from deep_dream import *
from helper_functions import *
import torchvision


# In[8]:


num_labels = 4000
model_name = "../models/net_4000_0.001_400_0.0.pth"
net = DictNet(num_labels)
net.load_state_dict(torch.load(model_name))


# In[9]:


dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)


# In[10]:


labels = np.random.choice(4000,200)
vals = np.zeros(np.size(labels))


# In[11]:


for index,label in enumerate(labels):
    img,val = dreamer(label=label)
    vals[index] = val


# In[12]:


np.histogram(vals,bins=10)
#plt.title("Number of labels correctly dreamt out of random 200")
#plt.savefig("hist_1.png")


# In[ ]:




