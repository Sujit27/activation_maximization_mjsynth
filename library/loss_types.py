from create_dream_loss_type import *

def main():

    loss_types = ['logit','softmax','log_softmax']
    
#    network = DictNet(1000)
#    network.load_state_dict(torch.load("../code/train_dict_network/out_3_1000/model_best.pth.tar")['state_dict'])
#    for loss_type in loss_types:
#        output_batch = dream(network,[80,366],(32,128,1),(0.47,),(0.14,),loss_type=loss_type)
#        save_image(output_batch,"mjsynth_"+loss_type+".png")
#
#    network = Net()
#    network.load_state_dict(torch.load('../deep_dream_mnist/mnist.pth'))
#    for loss_type in loss_types:
#        output_batch = dream(network,[3,4],(28,28,1),(0.13,),(0.31,),loss_type=loss_type)
#        save_image(output_batch,"mnist_" + loss_type +".png")
#
    network = models.vgg19(pretrained=True)
    for loss_type in loss_types:
        output_batch = dream(network,[71],(224,224,3),(0.485, 0.456, 0.406),(0.229, 0.224, 0.225),nItr=400,loss_type=loss_type)
        save_image(output_batch,"imageNet_" + loss_type +".png")


if __name__ == "__main__":
    main()
 
