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
from glue_layers import *
import torchvision
from timeit import default_timer as timer

NUM_SAMPLES = 3000
def main():
    num_labels = 500
    model_name = "../models/net_500_0.001_100_0.0.pth"
    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name))
    
    dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
    
    start = timer()
    for i in range(NUM_SAMPLES):
        random_seed = np.random.randint(1000)
        im,labels = dreamer.random_batch_dream(1,random_seed)
        
        words = label_to_word(labels)
        name = "_".join(elem for elem in words) + str(random_seed) + ".png"
        im = dreamer.show(im,name)
        plt.clf()
        plt.close()
        if i%(NUM_SAMPLES//10) == 0 and i!=0:
            print('{} dreams created'.format(i))
    stop = timer()
    print(stop-start)
    
if __name__ == "__main__":
    main()
    
