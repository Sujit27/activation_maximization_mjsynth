import sys
sys.path.append("../library/")
from dict_net import *
from helper_functions import *
import csv   



def train_model(output_path,transform,num_labels=None,lr=0.005,batch_size=16,weight_decay=0.001,num_epochs=1):
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #print("Device to be used {}".format(device))


    # create dataset
    #ds = dg.mjsynth.MjSynthWS('/mnt/c/Users/User/Desktop/mjsynth/')
    ds = dg.mjsynth.MjSynthWS('/var/tmp/on63ilaw/mjsynth/',transform)

    # subset the dataset with desired number of samples per label
    labels_indices_dict, ds = subset_dataset(ds,num_labels)
    #print("Number of labels: {}  Batch size: {}  Weight Decay factor: {}".format(num_labels,batch_size,weight_decay))
    
    # create a list of labels
    labels_list = list(labels_indices_dict)
    if num_labels is None:
        num_labels = len(labels_list)

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
    # create network with number of output nodes same as number of distinct labels
    net = DictNet2(num_labels)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    disp_batch_num_training = 2
    disp_batch_num_validation = 1
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
            targets = convert_target(targets,labels_list)

            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
#            running_loss += loss.item()
            if i % disp_batch_num_training == disp_batch_num_training-1:    # print every 50 mini-batches
#                print('[%d, %5d] training loss: %.3f' %
#                      (epoch + 1, i + 1, running_loss / disp_batch_num_training))
#                running_loss = 0.0
#            
                net.eval()
                with torch.no_grad():
                    images, targets = data
                    targets = convert_target(targets,labels_list)
                   
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    outputs = net(images)
                    
                    preds = one_hot_to_argmax(outputs)
                    training_acc_score = skm.accuracy_score(targets.cpu().detach().numpy(),preds.cpu().detach().numpy())
                    training_acc_score_list.append(training_acc_score)

#                    print("training accuracy_score : {}".format(training_acc_score))
                net.train()
        # break
#        if len(acc_score_list) > accuracy_num:
#            acc_last_n = acc_score_list[-accuracy_num:]
#            last_n_avg = sum(acc_last_n)/len(acc_last_n)
#            if last_n_avg > 0.9999999999:
#                break
#        if epoch % 20 == 19: 
#            saved_model_name = "checkpoint_"+str(num_labels) + "_"+str(num_samples_per_label)+"_"+str(epoch)+".pth"
#            torch.save({'model_state_dict':net.state_dict(),'optimizer_state_dict':optimizer.state_dict()},saved_model_name)
        net.eval()
        for j, data in enumerate(validationloader,0):
            with torch.no_grad():
                images, targets = data
                targets = convert_target(targets,labels_list)

                images = images.to(device)
                targets = targets.to(device)
                
                outputs = net(images)
                loss = criterion(outputs, targets)
                
                #validation_loss += loss.item()
                preds = one_hot_to_argmax(outputs)
                validation_acc_score = skm.accuracy_score(targets.cpu().detach().numpy(),preds.cpu().detach().numpy())                
                if i % disp_batch_num_validation == disp_batch_num_validation-1:    # print every 50 mini-batches
                    #                    print('[%d, %5d] validation loss: %.3f  validation accuracy score: %.3f' %
                    #                          (epoch + 1, j + 1, validation_loss / disp_batch_num_validation, validation_acc_score))
                    #                    validation_loss = 0.0
                    validation_acc_score_list.append(validation_acc_score)

        training_acc_avg = sum(training_acc_score_list)/len(training_acc_score_list)
        validation_acc_avg = sum(validation_acc_score_list)/len(validation_acc_score_list)
        print("{},{},{} ".format(epoch,training_acc_avg,validation_acc_avg))
        score_training_list.append(training_acc_avg)
        score_validation_list.append(validation_acc_avg)
    
    model_name = 'net_'+str(num_labels) + '_' + str(lr) +  '_' + str(batch_size) + '_' + str(weight_decay)+'.pth'
    output_file = os.path.join(output_path,model_name)
    torch.save(net.state_dict(),output_file)
    #print("MODEL SAVED"tter)
    return labels_list,score_training_list,score_validation_list

def main():
    output_path = "../models/"
    num_labels = 100
    lrs = [0.001]#[0.001,0.005,0.01]
    weight_decays = [0.00]
    batch_sizes = [64]
    num_epochs = 100
    transform = dg.mjsynth.mjsynth_gray_scale
    for batch_size in batch_sizes:
        for weight_decay in weight_decays:
            for lr in lrs:
                #print("#####")
                labels_list,score_training_list,score_validation_list = train_model(output_path,transform,num_labels=num_labels,lr = lr,batch_size=batch_size,weight_decay=weight_decay,num_epochs=num_epochs)
                file_name = "result_"+ str(num_labels) + "_l"+str(lr)+"_b"+str(batch_size)+"_w"+str(weight_decay)+".csv"
                output_csv = os.path.join(output_path,file_name)
                with open(output_csv,'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(score_training_list,score_validation_list))

                print("Result saved : {}".format(output_csv))

    labels_file = os.path.join(output_path,"labels.txt")
    with open(labels_file,'w') as f:
        for item in labels_list:
            f.write("%s\n" % item[1])

                
if __name__ == "__main__":
    main()
