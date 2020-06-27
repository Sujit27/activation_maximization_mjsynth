#!/bin/bash

#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.1_5_1 -o out_phocnet_3_1000_0.1_5_1   -ne 50 > out_train_result_3_1000_0.1_5_1.txt
#
#rm out_phocnet_3_1000_0.1_5_1/checkpoint.pth.tar
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.2_5_1 -o out_phocnet_3_1000_0.2_5_1   -ne 50 > out_train_result_3_1000_0.2_5_1.txt
#
#rm out_phocnet_3_1000_0.2_5_1/checkpoint.pth.tar
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.3_5_1 -o out_phocnet_3_1000_0.3_5_1   -ne 50 > out_train_result_3_1000_0.3_5_1.txt
#
#rm out_phocnet_3_1000_0.3_5_1/checkpoint.pth.tar
#
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_4_1000_0.1_5_1 -o out_phocnet_4_1000_0.1_5_1   -ne 50 > out_train_result_4_1000_0.1_5_1.txt
#
#rm out_phocnet_4_1000_0.1_5_1/checkpoint.pth.tar
#
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_5_1000_0.1_5_1 -o out_phocnet_5_1000_0.1_5_1   -ne 50 > out_train_result_5_1000_0.1_5_1.txt
#
#rm out_phocnet_5_1000_0.1_5_1/checkpoint.pth.tar
#
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_7_1000_0.1_5_1 -o out_phocnet_7_1000_0.1_5_1   -ne 50 > out_train_result_7_1000_0.1_5_1.txt
#
#rm out_phocnet_7_1000_0.1_5_1/checkpoint.pth.tar


python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.3_5_1 -o out_phocnet_all_5000_0.3_5_1   -ne 50 > out_train_result_all_5000_0.3_5_1.txt

#rm out_phocnet_all_5000_0.3_5_1/checkpoint.pth.tar

python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.5_5_1 -o out_phocnet_all_5000_0.5_5_1   -ne 50 > out_train_result_all_5000_0.5_5_1.txt

#rm out_phocnet_all_5000_0.5_5_1/checkpoint.pth.tar

python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.7_5_1 -o out_phocnet_all_5000_0.7_5_1   -ne 50 > out_train_result_all_5000_0.7_5_1.txt

#rm out_phocnet_all_5000_0.7_5_1/checkpoint.pth.tar

