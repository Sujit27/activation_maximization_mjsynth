import sys
sys.path.append("../library/")
from dict_net import *
from deep_dream import *
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m',type=str,default = '../models/net_1000_0.001_200_0.0.pth', dest='trained_model',help='Trained model for dreaming')
parser.add_argument('-l',type=int,default = 1, dest='label',help='Label to dream')
parser.add_argument('-r',type=int,default = 0, dest='random_seed',help='Random seed for creating the starting image for dreaming')
parser.add_argument('-itr',type=int,default = 500, dest='nItr',help='Number of iterations on dreaming')
parser.add_argument('-lr',type=float,default = 0.001, dest='lr',help='Learning rate')
parser.add_argument('-o',type=str,default = "../dreams/", dest='output_path',help='dream output location')


cmd_args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_softmaxed(net,image,transform):
    image = Variable(transform(image).unsqueeze(0))
    image = image.to(device)

    out = net(image)
    out_softmax = F.softmax(out,dim=1)
    value,index = out.max(1)
    prob,_ = out_softmax.max(1)

    return index[0],value[0],prob[0]



def main():
    #transform = dg.mjsynth.mjsynth_gray_pad
    trained_model = cmd_args.trained_model
    label = cmd_args.label
    random_seed = cmd_args.random_seed
    nItr = cmd_args.nItr
    lr = cmd_args.lr
    output_path = cmd_args.output_path

    filename = os.path.basename(trained_model)
    num_labels = int(filename.split('_')[1])
    if label < num_labels:
        net = DictNet(num_labels)
    else:
        print("Label number provided exceeds number of neurons in the last layer. Exiting...")
        return 1

    net.load_state_dict(torch.load(trained_model));
    dreamer = DeepDream(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)

    dream_im,val = dreamer(label=label,nItr=nItr,lr=lr,random_seed=random_seed)
    dream_im = dreamer.postProcess(dream_im)
    
    out_im_name = output_path+"dream_"+str(filename)+"_"+str(label)+"_"+str(nItr)+"_"+str(lr)+"_"+str(random_seed)+".png"
    dreamer.save(dream_im,out_im_name)

if __name__ == "__main__":
    main()
