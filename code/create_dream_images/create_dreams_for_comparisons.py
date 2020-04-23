import sys
sys.path.append("../library")
from dict_net import *
from deep_dream import *
from helper_functions import *
import torchvision
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-lr',type=bool,default = False, dest='change_lr',help='Change learning rate and dream')
parser.add_argument('-rs',type=bool,default = False, dest='change_random_seed',help='Change random seed of starting image')
parser.add_argument('-niter',type=bool,default = False, dest='change_nItr',help='Change number of iterations')
parser.add_argument('-sigma',type=bool,default = False, dest='change_sigma',help='Change sigma of gaussian filter')
parser.add_argument('-net',type=bool,default = False, dest='change_network',help='Change the network used for dreaming')
parser.add_argument('-tr',type=bool,default = False, dest='change_tr',help='Change type of training in the network')


cmd_args = parser.parse_args()


def main():
    lr_list = [0.001,0.01,0.05,0.1,0.5]
    rs_list = [0,1,2,3,4]
    nItr_list = [20,50,100,200,400]
    sigma_list = [0.3,0.4,0.5,0.6,0.7]
    model_list = ["../models/net_100_0.001_20_0.0.pth","../models/net_1000_0.001_200_0.0.pth","../models/net_2000_0.001_400_0.0.pth","../models/net_6000_0.001_200_0.0.pth"]
    train_type_dict = {"incrementally_trained":"../models/net_2000_0.001_200_0.0.pth","normally_trained":"../models/net_2000_0.001_400_0.0.pth"}

    lr = 0.1
    rs = 0
    nItr = 100
    sigma = 0.5
    model_name = "../models/net_1000_0.001_200_0.0.pth"
    output_path = "../dreams/"


    num_labels = int(model_name.split('_')[1])
    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name))
    dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
   
    #labels = np.random.choice(num_labels,4)
    labels = np.asarray([54,60,88,92])
    words = label_to_word(labels)

    if cmd_args.change_lr:
        for lr in lr_list:
            im = dreamer.batch_dream(labels=labels,lr=lr)

            name = "_".join(word for word in words) + '{:3.3f}'.format(lr) + ".png"
            name = os.path.join(output_path,name)
            dreamer.show(im,name)

    if cmd_args.change_random_seed:
        for rs in rs_list:
            im = dreamer.batch_dream(labels=labels,random_seed=rs)

            name = "_".join(word for word in words) + str(rs).zfill(3) + ".png"
            name = os.path.join(output_path,name)
            dreamer.show(im,name)

    if cmd_args.change_nItr:
        for nItr in nItr_list:
            im = dreamer.batch_dream(labels=labels,nItr=nItr)

            name = "_".join(word for word in words) + str(nItr).zfill(4)  + ".png"
            name = os.path.join(output_path,name)
            dreamer.show(im,name)

    if cmd_args.change_sigma:
        for sigma in sigma_list:
            dreamer.setGaussianFilter(sigma=sigma)
            im = dreamer.batch_dream(labels=labels)

            name = "_".join(word for word in words) + '{:3.3f}'.format(sigma) + ".png"
            name = os.path.join(output_path,name)
            dreamer.show(im,name)
    
    if cmd_args.change_network:
        for model_name in model_list:
            num_labels = int(model_name.split('_')[1])
            net = DictNet(num_labels)
            net.load_state_dict(torch.load(model_name))
            dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
           
            im = dreamer.batch_dream(labels=labels)

            name = "_".join(word for word in words) + str(num_labels).zfill(6) + ".png"
            name = os.path.join(output_path,name)
            dreamer.show(im,name)
    
    if cmd_args.change_tr:
        for training_type,model_name in train_type_dict.items():
            num_labels = int(model_name.split('_')[1])
            net = DictNet(num_labels)
            net.load_state_dict(torch.load(model_name))
            dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
           
            im = dreamer.batch_dream(labels=labels)

            name = "_".join(word for word in words) + "_" + training_type + ".png"
            name = os.path.join(output_path,name)
            dreamer.show(im,name)


    
if __name__ == "__main__":
    main()
    
