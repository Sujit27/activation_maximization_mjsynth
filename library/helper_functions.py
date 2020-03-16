from dict_net import *
import dagtasets as dg
import sklearn.metrics as skm
import csv
import ast


def convert_target(targets,labels_inv_dict):
    # converts targets from one given by getter of mjsynth dataset to one that
    # can be used by the neural net
    targets = targets.tolist()
    output = torch.zeros(len(targets),dtype=torch.long)
    for index_target,target in enumerate(targets):
        for label,label_num  in labels_inv_dict.items():
            if target == label:
                output[index_target] = label_num

    return output

def measure_accuracy(data,device,net,labels_inv_dict):
    images, targets = data
    targets = convert_target(targets,labels_inv_dict)
   
    images = images.to(device)
    targets = targets.to(device)
    
    outputs = net(images)
    
    preds = one_hot_to_argmax(outputs)
    acc_score = skm.accuracy_score(targets.cpu().detach().numpy(),preds.cpu().detach().numpy())

    return acc_score


def create_dicts(data_root,transform):
    # create dictionaries from csv files present in the library
    ds = dg.mjsynth.MjSynthWS(data_root,transform)
    labels_and_indices_dict =  csv_to_dict('../library/labels_and_indices.csv')
    labels_dict = csv_to_dict('../library/labels_1.csv')
    labels_inv_dict = csv_to_dict('../library/labels_2.csv')

    return ds, labels_and_indices_dict, labels_dict, labels_inv_dict



def label_to_word(label_num_list):
    labels_dict = csv_to_dict("../library/labels_full.csv")
    word_list = []
    for label_num in label_num_list:
        word_list.append(labels_dict[label_num][1])

    return word_list


# return argmax indices given a one hot encoded 2d tensor
def  one_hot_to_argmax(one_hot_output):  
    indices = np.zeros(len([*one_hot_output]))
    for i in range(one_hot_output.shape[0]):
        _,index= one_hot_output[i,:].max(0)       
        indices[i] = index
    
    indices = torch.from_numpy(indices).type(torch.LongTensor)
    
    return indices

def extract_name(filename):
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]
    word = " ".join(re.findall("[a-zA-Z]+",filename))
    word = word.lower()

    return word

def create_indices_list(labels_dict,ds):
    indices = []
    for key in labels_dict.keys():
        indices = indices + labels_dict[key]
    ds_new = torch.utils.data.Subset(ds,indices)
    
    return ds_new

def extract_dataset(ds,labels_and_indices_dict,labels_dict,num_labels,prev_num_labels=0):
    label_nums = [label_num for label_num in range(num_labels-prev_num_labels)]
    labels = [labels_dict[label_num] for label_num in label_nums]
    labels_dict = {}
    # choose only those labels from the labels_and_indices dict which need to be extractd
    for key,value in labels_and_indices_dict.items():
        if key[0] in labels:
            labels_dict[key] = value

    ds_new = create_indices_list(labels_dict,ds)

    return ds_new


def subset_dataset(ds,num_labels=None,num_samples_per_label=None):
    # set num_labels and num_samples_per_label to very high values if no arguements are passed
    if num_labels is None: num_labels = 1000000
    if num_samples_per_label is None: num_samples_per_label = 100000

    labels_dict = {}
    for i in range(len(ds)):
        class_id = ds.class_ids[i]
        class_name = extract_name(ds.filenames[i])
        label_key = (class_id,class_name)

        dict_length = len(labels_dict)
        if (dict_length < num_labels) and (label_key not in list(labels_dict)):
            labels_dict[label_key] = []
        
        if label_key in list(labels_dict):
            if (len(labels_dict[label_key]) < num_samples_per_label):
                labels_dict[label_key].append(i)

    ds_new = create_indices_list(labels_dict,ds)

    return labels_dict, ds_new

def subset_dataset_extend(ds,num_labels,prev_num_labels):
    # uses subset_dataset to return dataset that is added on top of a previously trained dataset
    labels_dict_small,_ = subset_dataset(ds,num_labels=prev_num_labels)
    labels_dict_large,_ = subset_dataset(ds,num_labels=num_labels)

    diff_dict = {i:labels_dict_large[i] for i in set(labels_dict_large) - set(labels_dict_small)}
    ds_new = create_indices_list(diff_dict,ds)
    
    return diff_dict,ds_new

def dict_to_csv(labels_dict,csv_file_name):
    with open(csv_file_name,'w',newline="") as csv_file:
        writer = csv.writer(csv_file)
        for key,value in labels_dict.items():
            writer.writerow([key, value])

def csv_to_dict(csv_file_name):
    with open(csv_file_name) as csv_file:
        reader = csv.reader(csv_file)
        label_dict = dict(reader)

    return literal_to_dict(label_dict)

def literal_to_dict(lit_dict):
    dictionary = {ast.literal_eval(key):ast.literal_eval(value) for key,value in lit_dict.items()}

    return dictionary

        
