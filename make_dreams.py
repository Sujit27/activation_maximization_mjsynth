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
    transform = dg.mjsynth.mjsynth_gray_pad
    model_dir = 'models_validated_reg/'
    output_path = "dreams/"
    saved_networks = ["net_5_None"]
    
    nItrs = [100,200,400,800]
    lrs = [0.1] #[0.001,0.005,0.01,0.05,0.1,0.5]
    label = 1
    for saved_network in saved_networks:
        filename = os.path.basename(saved_network)
        num_labels = int(filename.split("_")[1])
        net = DictNet(num_labels)

        net.load_state_dict(torch.load(model_dir + saved_network+".pth"));
        dreamer = DeepDream(net)

        for nItr in nItrs:
            for lr in lrs:
                dream_im = dreamer(label=label,nItr=nItr,lr=lr)
    #            dream_im = dreamer.createInputImage()
    #            dream_im = dreamer.prepInputImage(dream_im)
                dream_im = dreamer.postProcess(dream_im)
                
                label_predicted,activation,probability = label_softmaxed(dreamer.net,dream_im,transform)
                print("Label predicted : {} Activation predicted : {} Probability predicted : {} ".format(label_predicted,activation,probability))
 
#                label_predicted,activation = label_softmaxed(dreamer.net,dream_im,transform)
#                print("Label predicted : {} Probability predicted : {}".format(label_predicted,activation))
                out_im_name = output_path+"dream_"+str(filename)+"_"+str(label)+"_"+str(nItr)+"_"+str(lr)+".png"
                dreamer.save(dream_im,out_im_name)

if __name__ == "__main__":
    main()
