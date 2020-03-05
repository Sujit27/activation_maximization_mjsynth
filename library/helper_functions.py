from dict_net import *
import csv
import ast


def convert_target(targets,labels_inv_dict):
    targets = targets.tolist()
    output = torch.zeros(len(targets),dtype=torch.long)
    for index_target,target in enumerate(targets):
        for label,label_num  in labels_inv_dict.items():
            if target == label:
                output[index_target] = label_num

    return output

def label_to_word(label_list):
    labels_and_indices_dict = csv_to_dict("labels_and_indices.csv")
    label_dict = csv_to_dict("labels_1.csv")
    labels = [label_dict[label_num] for label_num in label_list]
    word_list = [key[1] for key in labels_and_indices_dict.keys() if key[0] in labels]

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

def main():
    ds = dg.mjsynth.MjSynthWS('/var/tmp/on63ilaw/mjsynth/',dg.mjsynth.mjsynth_gray_scale)
    labels_indices_dict, _ = subset_dataset(ds)
    csv_file_name1 = 'labels_and_indices.csv'
    dict_to_csv(labels_indices_dict,csv_file_name1)

    print("Labels and indices csv file saved")

    labels_list = list(labels_indices_dict)
    labels_map_full = {}
    labels_map1 = {}
    labels_map2 = {}
    for index,value in enumerate(labels_list):
        labels_map_full[index] = value
        labels_map1[index] = value[0]
        labels_map2[value[0]] = index

    csv_file_name2 = 'labels_full.csv'
    csv_file_name3 = 'labels_1.csv'
    csv_file_name4 = 'labels_2.csv'
    dict_to_csv(labels_map_full,csv_file_name2)
    dict_to_csv(labels_map1,csv_file_name3)
    dict_to_csv(labels_map2,csv_file_name4)
    print("Labels files saved")
 

if __name__ == "__main__":
    main()
        
