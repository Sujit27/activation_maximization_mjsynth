import sys
sys.path.append("../../")
import dagtasets as dg
from library.dict_network.dict_net import *
from library.helper_functions import *
from library.label_dicts import create_label_dicts
from torch.utils.data  import SubsetRandomSampler
from torch import optim
import csv
import os
import glob
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n',type=int,default = None, dest='num_labels',help='Store number of labels')
parser.add_argument('-m',type=str,default = None, dest='existing_model_location',help='Full path of existing model')
parser.add_argument('--grow',default = False, action = 'store_true',help='Bool whether to grow the existing model')
parser.add_argument('--dicts',default = False, action = 'store_true',help='Bool whether to create fresh label indices dictionary at the data root. If training on the same data as previous, it saves time to set this False')
parser.add_argument('-o',type=str,default = "out/", dest='output_path',help='Output model location')
parser.add_argument('-d',type=str,default = "/var/tmp/on63ilaw/mjsynth/", dest='data_root',help='input data location')
parser.add_argument('-lr',type=float,default = 0.001, dest='lr',help='Learning rate')
parser.add_argument('-ne',type=int,default = 30, dest='num_epochs',help='Number of epochs to train')
parser.add_argument('--nice',default = True, action = 'store_true',help='Bool whether to nice')


cmd_args = parser.parse_args()


def train_model(grow_prev_model,output_path,data_root,transform,prev_trained_model_name=None,num_labels=None,lr=0.005,batch_size=16,weight_decay=0.001,num_epochs=1):
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    ds, labels_and_indices_dict, labels_dict, labels_inv_dict = create_dicts(data_root,transform)

    # create network with number of output nodes same as number of distinct labels
    if grow_prev_model == False:
        net = DictNet(num_labels)
        # subset the dataset with desired number of samples per label
        ds = extract_dataset(ds,labels_and_indices_dict,labels_dict,num_labels,prev_num_labels=0)
        if prev_trained_model_name is not None:
            net.load_state_dict(torch.load(prev_trained_model_name))
 
    else:
        prev_num_labels = int(((os.path.basename(prev_trained_model_name)).split("_"))[1])
        trained_net = DictNet(prev_num_labels)
        trained_net.load_state_dict(torch.load(prev_trained_model_name))
        net = grow_net(trained_net,num_labels)
        
        # subset the dataset with desired number of samples per label
        ds = extract_dataset(ds,labels_and_indices_dict,labels_dict,num_labels,prev_num_labels)


    net.to(device)

    # splot training and validation data
    validation_split = 0.15
    dataset_size = len(ds)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    shuffle_dataset = True
    #random_seed= np.random.randint(100)
    random_seed = 0

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # create a trainloader for the data subset
    trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,sampler=train_sampler) 
    validationloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,sampler=valid_sampler)

    #trainloader = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    score_training_list = []
    score_validation_list = []
    for epoch in range(num_epochs):
        net.train()
        #running_loss = 0.0
        #validation_loss = 0.0
        training_acc_score_list = []
        validation_acc_score_list = []
        for i, data in enumerate(trainloader, 0):
            images, targets = data
            # return position of the targets in labels_list as the new target
            targets = convert_target(targets,labels_inv_dict)

            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # evaluate training data
            net.eval()
            with torch.no_grad():
                training_acc_score_list.append(measure_accuracy(data,device,net,labels_inv_dict))

#                    print("training accuracy_score : {}".format(training_acc_score))
            net.train()
        
        # evaluate validation data
        net.eval()
        for j, data in enumerate(validationloader,0):
            with torch.no_grad():
                validation_acc_score_list.append(measure_accuracy(data,device,net,labels_inv_dict))

        training_acc_avg = sum(training_acc_score_list)/len(training_acc_score_list)
        validation_acc_avg = sum(validation_acc_score_list)/len(validation_acc_score_list)
        print("{},{},{} ".format(epoch,training_acc_avg,validation_acc_avg))
        score_training_list.append(training_acc_avg)
        score_validation_list.append(validation_acc_avg)
    
    model_name = 'net_'+str(num_labels) + '_' + str(lr) +  '_' + str(batch_size) + '_' + str(weight_decay)+'.pth'
    output_file = os.path.join(output_path,model_name)
    torch.save(net.state_dict(),output_file)

    # delete network and empty cache
    del net
    torch.cuda.empty_cache()
    #print("MODEL SAVED"tter)
    return score_training_list,score_validation_list

def main():
    print(cmd_args)
    # Can either train a model from start given the number of labels 
    # Or can grow an existing trained model ( see library/dict_network for more details ) by increasing the number of labels to a specified number 
    num_labels = cmd_args.num_labels
    prev_trained_model_name = cmd_args.existing_model_location
    grow_prev_model = cmd_args.grow
    output_path = cmd_args.output_path
    data_root = cmd_args.data_root
    lr = cmd_args.lr
    num_epochs = cmd_args.num_epochs
    nice = cmd_args.nice

    prev_num_labels = None
    if prev_trained_model_name is not None:
        prev_num_labels = int(((os.path.basename(prev_trained_model_name)).split("_"))[1])
        if prev_num_labels > num_labels:
            print("Number of labels requested is lesser than the number of labels in the previously trained model. Qutting ...")
            return
        if grow_prev_model:
            if prev_num_labels > num_labels:
                print("Number of labels requested is lesser than the number of labels in the previously trained model.Cannot grow. Qutting ...")
                return

    if nice is True : 
        os.nice(20)

    weight_decay = 0.00
    transform = dg.mjsynth.mjsynth_gray_scale
            
    if grow_prev_model:
        batch_size = int((num_labels-prev_num_labels)/5)
    else:
        batch_size = int(num_labels/5) 
    batch_size = min(batch_size,256)

    # create dictionary csv of labels and indices at the data root 
    if cmd_args.dicts is True:
        for csvpath in glob.iglob(os.path.join(data_root,'*.csv')):
            os.remove(csvpath)
        create_label_dicts(data_root)

    # create output directory for saving model if does not exist already
    Path(output_path).mkdir(parents=True,exist_ok=True)

    # train
    score_training_list,score_validation_list = train_model(grow_prev_model,output_path,data_root,transform,prev_trained_model_name=prev_trained_model_name,num_labels=num_labels,lr = lr,batch_size=batch_size,weight_decay=weight_decay,num_epochs=num_epochs)
    
    # save csv
    file_name = "result_"+ str(num_labels) + "_l"+str(lr)+"_b"+str(batch_size)+"_w"+str(weight_decay)+".csv"
    output_csv = os.path.join(output_path,file_name)
    with open(output_csv,'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(score_training_list,score_validation_list))

    print("Result saved : {}".format(output_csv))
    
            
                
if __name__ == "__main__":
    main()
