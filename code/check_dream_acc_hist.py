#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
sys.path.append("../library")
from dict_net import *
from deep_dream import *
from helper_functions import *
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m',type=str,default = '../models/net_1000_0.001_200_0.0.pth', dest='model_name',help='Trained model for dreaming')
parser.add_argument('-n',type=int,default = 200, dest='num_samples',help='Number of random samples to dream')

cmd_args = parser.parse_args()

def main():
    model_name = cmd_args.model_name
    num_samples = cmd_args.num_samples
    filename = os.path.basename(model_name)
    num_labels = int(filename.split('_')[1])

    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name))

    dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
    labels = np.random.choice(num_labels,num_samples)
    vals = np.zeros(np.size(labels))

    for index,label in enumerate(labels):
        img,val = dreamer(label=label)
        vals[index] = val

    hist = np.histogram(vals,bins=10)
    print(hist)

if __name__ == "__main__":
    main()



