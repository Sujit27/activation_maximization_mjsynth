# coding: utf-8
import sys
sys.path.append("../library")
sys.path.append("../library/SR_GAN/")
from deep_dream import *
from models import *
from dict_net import *
from helper_functions import *

import os
import glob

gen_models_loc = "../models_generator/"
model = "../models2/net_1000_0.001_200_0.0.pth"


device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr,rs,nItr,sigma = 0.1,0,100,0.5
net = DictNet(1000)
net.load_state_dict(torch.load(model))
dreamer = DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
labels = np.asarray([54,60,88,92])
words = label_to_word(labels)

gen_models = glob.glob(os.path.join(gen_models_loc,"*.pth"))
print("Generator models\n",gen_models)

generator = Generator()

im  = dreamer.batch_dream(labels=labels)
name = "_".join(word for word in words) + ".png"
name = os.path.join(gen_models_loc,name)
dreamer.show(im,name)

for i in range(len(gen_models)):
    im  = dreamer.batch_dream(labels=labels)
    generator.load_state_dict(torch.load(gen_models[i]))
    generator.to(device)
    im2 = generator(im)

    name = "_".join(word for word in words) + str(i) + ".png"
    name = os.path.join(gen_models_loc,name)
    dreamer.show(im2,name)
