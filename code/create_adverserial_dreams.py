import sys
sys.path.append("../library")
from deep_dream import *
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-m',type=str,default='../models/net_1000_0.001_200_0.0.pth',dest='model',help='Model location for dreamer')
parser.add_argument('-d',type=str,default=None,dest='discrim_model',help='Model location for discriminator')
parser.add_argument('-nAdvL',type=int,default=10,dest='n_adv_loops',help='number of adversarial loops')
parser.add_argument('-nItrD',type=int,default=10,dest='nItr_d',help='number of discriminator iterations in one loop')
parser.add_argument('-lrD',type=float,default=0.1,dest='lr_d',help='learning rate of discriminator for adversarial dreaming')
parser.add_argument('-plain',type=bool,default=False,dest='plain_dream',help='bool for plain dreaming')

cmd_args = parser.parse_args()

labels = [101,201,301,401,501,601,701,801]

def main():
    model = cmd_args.model
    discrim_model = cmd_args.discrim_model
    n_adv_loops = cmd_args.n_adv_loops
    nItr_d = cmd_args.nItr_d
    lr_d = cmd_args.lr_d
    plain_dream = cmd_args.plain_dream

    num_labels = int(model.split('_')[1])
    net = DictNet(num_labels)
    net.load_state_dict(torch.load(model))
    
    if discrim_model is not None:
        discrim_net = DiscrimNet()
        discrim_net.load_state_dict(torch.load(discrim_model))
        print("Discriminator network initialized with:",discrim_model)
    else:
        discrim_net = None
        print("Discriminator network randomly initialized")

    dreamer = DeepDreamGAN(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True,discrim_net=discrim_net)
    if plain_dream:
        print("Dreaming without adversarial")
        dream_tensor = dreamer.batch_dream(labels)
    else:
        print("Dreaming adversarially")
        dream_tensor = dreamer.batch_dream_GAN(labels,n_adv_loops=n_adv_loops,nItr_d=nItr_d,lr_d=lr_d)


    words = label_to_word(labels)
    image_name = "_".join(words) + ".png"
    print(dream_tensor.shape)
    dreamer.show(dream_tensor,image_name)

if __name__ == "__main__":
    main()
