import sys
sys.path.append("../library/")
from dict_net import *
from helper_functions import *
import csv
import ast
import argparse

# creates dictopnaries  of labels as csv in library directory that will be used for training
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d',type=str,default = "/var/tmp/on63ilaw/mjsynth/", dest='data_loc',help='data location')
parser.add_argument('-s',type=str,default = "../library/", dest='save_loc',help='saving location for dictionaries')


cmd_args = parser.parse_args()


def main():
    data_loc = cmd_args.data_loc
    save_loc = cmd_args.save_loc
    ds = dg.mjsynth.MjSynthWS(data_loc,dg.mjsynth.mjsynth_gray_scale)
    labels_indices_dict, _ = subset_dataset(ds)
    csv_file_name1 = 'labels_and_indices.csv'
    csv_file_name1 = os.path.join(save_loc,csv_file_name1)
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
    csv_file_name2 = os.path.join(save_loc,csv_file_name2)
    csv_file_name3 = os.path.join(save_loc,csv_file_name3)
    csv_file_name4 = os.path.join(save_loc,csv_file_name4)
    dict_to_csv(labels_map_full,csv_file_name2)
    dict_to_csv(labels_map1,csv_file_name3)
    dict_to_csv(labels_map2,csv_file_name4)
    print("Labels files saved")
 

if __name__ == "__main__":
    main()
 
