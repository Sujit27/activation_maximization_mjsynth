import sys
sys.path.append("../library")
from dict_net import *
from deep_dream import *
from helper_functions import *
import torchvision
from timeit import default_timer as timer
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = '1000', dest='num_samples',help='Number of dream samples to be generated')
parser.add_argument('-m',type=str,default = '../models/net_1000_0.001_200_0.0.pth', dest='model_name',help='Trained model for dreaming')
parser.add_argument('-o',type=str,default = "../dreams/", dest='output_path',help='dream output location')


cmd_args = parser.parse_args()


def main():
    num_samples = cmd_args.num_samples
    model_name = cmd_args.model_name
    output_path = cmd_args.output_path
    num_labels = int(model_name.split('_')[1])

    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model_name))
    
    dreamer= DeepDreamBatch(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)
    
    start = timer()
    for i in range(num_samples):
        random_seed = np.random.randint(1000)
        im,labels = dreamer.random_batch_dream(1,random_seed)
        
        words = label_to_word(labels)
        name = "_".join(elem for elem in words) + str(random_seed) + ".png"
        name = os.path.join(output_path,name)
        dreamer.show(im,name)
        if i%100 == 99 :
            print('{} dreams created'.format(i))
    stop = timer()
    time_duration = stop - start
    print("Total time taken : {}".format(time_duration))
    
if __name__ == "__main__":
    main()
    
