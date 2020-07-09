#!/bin/bash

# train phocnet length 3, num of labels 1000, different splits
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.5_5_1 -o out_phocnet_3_1000_0.5_5_1   -ne 50 > out_train_result_3_1000_0.5_5_1.txt

#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.7_5_1 -o out_phocnet_3_1000_0.7_5_1   -ne 50 > out_train_result_3_1000_0.7_5_1.txt

#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.9_5_1 -o out_phocnet_3_1000_0.9_5_1   -ne 50 > out_train_result_3_1000_0.9_5_1.txt
#
#
## train phocnet, lengths different, split 0.5
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_4_1000_0.5_5_1 -o out_phocnet_4_1000_0.5_5_1   -ne 50 > out_train_result_4_1000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_5_1000_0.5_5_1 -o out_phocnet_5_1000_0.5_5_1   -ne 50 > out_train_result_5_1000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_6_1000_0.5_5_1 -o out_phocnet_6_1000_0.5_5_1   -ne 50 > out_train_result_6_1000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_7_1000_0.5_5_1 -o out_phocnet_7_1000_0.5_5_1   -ne 50 > out_train_result_7_1000_0.5_5_1.txt
#
#
## train phocnet all lengths,0.5 split, different number of labels
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_1000_0.5_5_1 -o out_phocnet_all_1000_0.5_5_1   -ne 50 > out_train_result_all_1000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_2000_0.5_5_1 -o out_phocnet_all_2000_0.5_5_1   -ne 50 > out_train_result_all_2000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.5_5_1 -o out_phocnet_all_5000_0.5_5_1   -ne 50 > out_train_result_all_5000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_8000_0.5_5_1 -o out_phocnet_all_8000_0.5_5_1   -ne 50 > out_train_result_all_8000_0.5_5_1.txt
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_10000_0.5_5_1 -o out_phocnet_all_10000_0.5_5_1   -ne 50 > out_train_result_all_10000_0.5_5_1.txt
#
#
## train phocnet of all lengths, num of labels 5000, diffrent splits
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.1_5_1 -o out_phocnet_all_5000_0.1_5_1   -ne 50 > out_train_result_all_5000_0.1_5_1.txt
python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.7_5_1 -o out_phocnet_all_5000_0.7_5_1   -ne 50 > out_train_result_all_5000_0.7_5_1.txt
python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.9_5_1 -o out_phocnet_all_5000_0.9_5_1   -ne 50 > out_train_result_all_5000_0.9_5_1.txt
python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.3_5_1 -o out_phocnet_all_5000_0.3_5_1   -ne 50 > out_train_result_all_5000_0.3_5_1.txt
