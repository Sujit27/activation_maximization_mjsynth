from dict_net import *

## labels list to one hot encoding
#def list_to_one_hot_dict(labels):
#    distinct_words = set(labels)
#    distinct_words = list(distinct_words)
#    distinct_words.sort()
#    num_labels = len(distinct_words)
#    one_hot_dict = {}
#    index = 0
#    for word in distinct_words:
#        one_hot_dict[word] = np.zeros(num_labels)
#        one_hot_dict[word][index] = 1.0
#        index += 1 
#    return one_hot_dict
#
## return target for training/validation
#def one_hot_dict_encode(targets,one_hot_dict):
#    num_labels = len(list(one_hot_dict.values())[0])
#    one_hot_output = np.zeros((len(targets),num_labels))
#    for index in range(len(targets)):
#        one_hot_output[index,:] = one_hot_dict[targets[index].item()]
#    
#    one_hot_output = torch.from_numpy(one_hot_output)
#
#    return one_hot_to_argmax(one_hot_output)
#
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

#def subset_dataset(ds,num_samples_per_label):
#    indices = []
#    class_ids = []
#    class_names = []
#    for i in range(len(ds)):
#        
#        if (class_ids.count(ds.class_ids[i]) < num_samples_per_label) :
#            indices.append(i)
#            class_ids.append(ds.class_ids[i])
#            class_names.append(extract_name(ds.filenames[i]))
#
#
#    ds_new = torch.utils.data.Subset(ds,indices)
#    
#    return class_names,class_ids,ds_new
#
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

    indices = []
    for key in labels_dict.keys():
        indices = indices + labels_dict[key]
    ds_new = torch.utils.data.Subset(ds,indices)
    
    return labels_dict, ds_new
 
