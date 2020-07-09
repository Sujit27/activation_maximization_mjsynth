#!/bin/bash

# train phocnet on known dream datasets
python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_3_500_known -o out_phocnet_3_500_known   -ne 50 > out_train_result_3_500_known.txt

python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_7_2000_known -o out_phocnet_7_2000_known   -ne 50 > out_train_result_7_2000_known.txt

python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_known -o out_phocnet_all_5000_known   -ne 50 > out_train_result_all_5000_known.txt


