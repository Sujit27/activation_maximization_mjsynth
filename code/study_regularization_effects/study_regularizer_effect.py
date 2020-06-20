import sys
sys.path.append("../library")

from create_dream_reg import *

def main():

    network = DictNet(1000)
    network.load_state_dict(torch.load("train_dict_network/out5/net_1000_0.001_200_0.0.pth",map_location=torch.device('cpu')))
    
    outpath = "out_reg_dreams_sample"    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    #filters = ['box','gauss','median']
    filters = ['gauss','median']
    filter_freqs = [1,5,10,20]
    kernel_sizes = [3,5,9]
    sigmas = [0.3,0.5,0.7]
    
    for filter in filters:
        for filter_freq in filter_freqs:
            for kernel_size in kernel_sizes:
                for sigma in sigmas:
                    if filter != 'gauss':
                        regularizer = [(filter,filter_freq),kernel_size]
                        output_batch = dream(network,[0,1,2,3],(32,128,1),(0.47,),(0.14,),regularizer=regularizer)
                        file_name = os.path.join(outpath, "dreams_mjsynth_" + filter + "_" + str(filter_freq) + "_" + str(kernel_size) + ".png")
                        save_image(output_batch,file_name)
                        print("File saved :",file_name)
                        break
                    elif filter == 'gauss':
                        regularizer = [(filter,filter_freq),kernel_size,sigma]
                        output_batch = dream(network,[0,1,2,3],(32,128,1),(0.47,),(0.14,),regularizer=regularizer)
                        file_name = os.path.join(outpath, "dreams_mjsynth_" + filter + "_" + str(filter_freq) + "_" + str(kernel_size) + "_" + str(sigma) + ".png")
                        save_image(output_batch,file_name)
                        print("File saved :",file_name)
                        
                    
if __name__ == "__main__":
    main()