import sys
sys.path.append("../library/")
from dict_net import *
from deep_dream import *


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
    if len(sys.argv) < 3:
        print(len(sys.argv))
        print("Add name of the model and number of lables as command line arguements")
        return 1
    output_path = "../dreams/"
    trained_model = sys.argv[1]
    label = int(sys.argv[2])
    if len(sys.argv) == 3:
        random_seed = 0
    else:
        random_seed = int(sys.argv[3])
    
    nItrs = [400]
    lrs = [0.1] #[0.001,0.005,0.01,0.05,0.1,0.5]
    filename = os.path.basename(trained_model)
    num_labels = int(filename.split('_')[1])
    if label < num_labels:
        net = DictNet(num_labels)
    else:
        print("Label number provided exceeds number of neurons in the last layer. Exiting...")
        return 1

    net.load_state_dict(torch.load(trained_model));
    dreamer = DeepDream(net,(1,32,128),(0.47,),(0.14,),use_gaussian_filter=True)

    for nItr in nItrs:
        for lr in lrs:
            dream_im = dreamer(label=label,nItr=nItr,lr=lr,random_seed=random_seed)
#            dream_im = dreamer.createInputImage()
#            dream_im = dreamer.prepInputImage(dream_im)
            dream_im = dreamer.postProcess(dream_im)
            
            #label_predicted,activation,probability = label_softmaxed(dreamer.net,dream_im,transform)
            #print("Label predicted : {} Activation predicted : {} Probability predicted : {} ".format(label_predicted,activation,probability))

#                label_predicted,activation = label_softmaxed(dreamer.net,dream_im,transform)
#                print("Label predicted : {} Probability predicted : {}".format(label_predicted,activation))
            out_im_name = output_path+"dream_"+str(filename)+"_"+str(label)+"_"+str(nItr)+"_"+str(lr)+"_"+str(random_seed)+".png"
            dreamer.save(dream_im,out_im_name)

if __name__ == "__main__":
    main()
