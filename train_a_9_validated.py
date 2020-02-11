from dict_net import *
from helper_functions import *
   



def train_model(transform,num_labels=None,num_samples_per_label=None,num_epochs=1):
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device to be used {}".format(device))


    # create dataset
    #ds = dg.mjsynth.MjSynthWS('/mnt/c/Users/User/Desktop/mjsynth/')
    ds = dg.mjsynth.MjSynthWS('/var/tmp/on63ilaw/mjsynth/',transform)

    # subset the dataset with desired number of samples per label
    labels_indices_dict, ds = subset_dataset(ds,num_labels)
    print("Number of labels: {} \n  Number of samples per label: {}".format(num_labels,num_samples_per_label))
    
    # create a list of labels
    labels_list = list(labels_indices_dict)

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

    batch_size = 16
    lr = 0.005
    weight_decay = 0.001
    # create a trainloader for the data subset
    trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,sampler=train_sampler) 
    validationloader = torch.utils.data.DataLoader(ds, batch_size=batch_size,sampler=valid_sampler)
    #trainloader = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True)
    # create network with number of output nodes same as number of distinct labels
    net = DictNet(num_labels)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    disp_batch_num_training = 5
    disp_batch_num_validation = 1
    validation_acc_score_list = []
    accuracy_num = 20 # variables that say how many last accuracy scores to look at for stopping training
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        validation_loss = 0.0
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
            
            running_loss += loss.item()
            if i % disp_batch_num_training == disp_batch_num_training-1:    # print every 50 mini-batches
                print('[%d, %5d] training loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / disp_batch_num_training))
                running_loss = 0.0
            
                net.eval()
                with torch.no_grad():
                    images, targets = data
                    targets = convert_target(targets,labels_list)
                   
                    images = images.to(device)
                    targets = targets.to(device)
                    
                    outputs = net(images)
                    
                    preds = one_hot_to_argmax(outputs)
                    training_acc_score = skm.accuracy_score(targets.cpu().detach().numpy(),preds.cpu().detach().numpy())                

                    print("training accuracy_score : {}".format(training_acc_score))
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
                
                validation_loss += loss.item()
                preds = one_hot_to_argmax(outputs)
                validation_acc_score = skm.accuracy_score(targets.cpu().detach().numpy(),preds.cpu().detach().numpy())                
                if i % disp_batch_num_validation == disp_batch_num_validation-1:    # print every 50 mini-batches
                    print('[%d, %5d] validation loss: %.3f  validation accuracy score: %.3f' %
                          (epoch + 1, j + 1, validation_loss / disp_batch_num_validation, validation_acc_score))
                    validation_loss = 0.0
                    validation_acc_score_list.append(validation_acc_score)
             
    torch.save(net.state_dict(),'net_'+str(num_labels) + '_' + str(num_samples_per_label)+'.pth')
    print("MODEL SAVED")
    return labels_indices_dict,labels_list

def main():
    num_labels_list = [5]
    num_samples_per_labels = [None]
    num_epochs = 500
    transform = dg.mjsynth.mjsynth_gray_pad
    for num_labels in num_labels_list:
        for num_samples_per_label in num_samples_per_labels:
            print("#####")
            labels_indices_dict,labels_list = train_model(transform,num_labels,num_samples_per_label,num_epochs)
            print("Labels Indices Dictionary : {}".format(labels_indices_dict))
            print("Label List : {}".format(labels_list))
if __name__ == "__main__":
    main()
