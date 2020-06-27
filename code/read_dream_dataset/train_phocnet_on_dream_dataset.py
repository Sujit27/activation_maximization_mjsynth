import sys
sys.path.append("../../library")
sys.path.append("../../library/phoc_network")
import shutil
import argparse
import time
from train_phocnet_on_dream_images import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d',type=str,default = None, dest='data_path',required=True,help='Location of training and test data. Should have directories called train and test') 
parser.add_argument('-o',type=str,default = 'out', dest='output_path',help='Location where the phocnet will be saved') 
parser.add_argument('-bs',type=int,default = 64, dest='batch_size',help='batch size for training phocnet')
parser.add_argument('-ne',type=int,default = 100, dest='num_epochs',help='number of epochs for training phocnet')
parser.add_argument('-lr',type=float,default = 0.0001, dest='learning_rate',help='learning rate for training phocnet')

args = parser.parse_args()

def main():
    training_data_path = os.path.join(args.data_path,'train')
    test_data_path =  os.path.join(args.data_path,'test')

    phoc_pooling_levels = [1,2,3,4,5,6,7,8,9,10]

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    start = time.time()

    train_phocNet_on_dream_dataset(phoc_pooling_levels,training_data_path,args.output_path,args.num_epochs,args.learning_rate,args.batch_size)

    end = time.time()

    print("Time taken for training and testing of phocNet on current set of dream images is : {}".format(end-start))

if __name__ == "__main__":
    main()
