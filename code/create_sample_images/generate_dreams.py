import sys
sys.path.append("../../library")
import argparse
import random

from dict_network.dict_net import *
from create_dream import *
from helper_functions import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m',type=str,default = None, dest='trained_net_dir',required=True, help='Directory where trained dictnet is present. Trained model inside should be named model_best.pth.tar and dictionary of classes inside should be named label_dict.json')
parser.add_argument('-n',type=int,default = None, dest='num_labels',help='Number of labels in the trained dictnet')
parser.add_argument('--labels', nargs='+', type=int,default=None,help='List of labels on which to dream')
parser.add_argument('-o',type=str,default = "out", dest='output_path',help='dream output location')

cmd_args = parser.parse_args()


def generate_dreams(net,label_dict,labels,output_path):
    output = dream(net,labels,(32,128,1),(0.47,),(0.14,))
    save_images(output,labels,label_dict,output_path,0)

def main():
    if cmd_args.labels is None:
        random.seed(0)
        labels = random.sample(range(cmd_args.num_labels),4)
    else:
        labels = cmd_args.labels

    net = DictNet(cmd_args.num_labels)
    net.load_state_dict(torch.load(os.path.join(cmd_args.trained_net_dir,'model_best.pth.tar'))['state_dict'])

    dict_file = os.path.join(cmd_args.trained_net_dir,'label_dict.json')
    with open(dict_file) as json_file:
        label_dict = json.load(json_file)
    
    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)

    generate_dreams(net,label_dict,labels,cmd_args.output_path)

if __name__ == "__main__":
    main()
    

