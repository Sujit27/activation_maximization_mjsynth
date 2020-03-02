from dict_net import *


def convert_target(targets,labels_list):
    targets = targets.tolist()
    output = torch.zeros(len(targets),dtype=torch.long)
    for index_target,item in enumerate(targets):
        for index_label,value in enumerate(labels_list):
            if item == value[0]:
                output[index_target] = index_label

    return output

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
