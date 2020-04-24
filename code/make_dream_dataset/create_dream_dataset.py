import sys
sys.path.append("../../")
from library.dict_network.dict_net import *
from library.deep_dream import *
from library.helper_functions import *
import torchvision
from timeit import default_timer as timer
import argparse
import os
import glob
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = '1000', dest='num_samples',help='Number of dream samples to be generated')
parser.add_argument('-m',type=str,default = None, dest='model_name',help='Trained model for dreaming')
parser.add_argument('-o',type=str,default = "out", dest='output_path',help='dream output location')


cmd_args = parser.parse_args()

def create_annotation_txt(files_path):
    '''
    creates a txt file listing the names of all the jpg files
    in the given directory
    '''
    file_list = glob.glob(os.path.join(files_path,"*.jpg"))
    output_file_name = os.path.join(files_path,"annotation_train.txt")

    print("Creating annotation txt file")

    with open(output_file_name,'w') as f:
        for item in file_list:
            f.write("%s\n" % os.path.basename(item))

def main():
    num_samples = cmd_args.num_samples
    model_name = cmd_args.model_name
    output_path = cmd_args.output_path

    model_basename = os.path.basename(model_name)
    num_labels = int(model_basename.split('_')[1])

    Path(output_path).mkdir(parents=True,exist_ok=True)

    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name))
    
    dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
    
    for i in range(num_samples):
        random_seed = np.random.randint(1000)
        im,labels = dreamer.random_batch_dream(1,random_seed)
        
        words = label_to_word(labels)
        name = str(i) + "_" + words[0] + "_" + str(random_seed) + ".jpg"
        name = os.path.join(output_path,name)
        dreamer.show(im,name)
        if i%100 == 99 :
            print('{} dreams created'.format(i))

    create_annotation_txt(output_path)
    
if __name__ == "__main__":
    main()
    
