#!/bin/bash

#histograms for length 3, num of labels 1000, different splits
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.1_5_1/ -m out_phocnet_3_1000_0.1_5_1/model_best.pth.tar -hsg l_3_n_1000_s_0.1
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.3_5_1/ -m out_phocnet_3_1000_0.3_5_1/model_best.pth.tar -hsg l_3_n_1000_s_0.3
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.5_5_1/ -m out_phocnet_3_1000_0.5_5_1/model_best.pth.tar -hsg l_3_n_1000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.7_5_1/ -m out_phocnet_3_1000_0.7_5_1/model_best.pth.tar -hsg l_3_n_1000_s_0.7
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_3_1000_0.9_5_1/ -m out_phocnet_3_1000_0.9_5_1/model_best.pth.tar -hsg l_3_n_1000_s_0.9


#histogram of different lengths,num of labels 1000, 0.5 split
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_4_1000_0.5_5_1/ -m out_phocnet_4_1000_0.5_5_1/model_best.pth.tar -hsg l_4_n_1000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_5_1000_0.5_5_1/ -m out_phocnet_5_1000_0.5_5_1/model_best.pth.tar -hsg l_5_n_1000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_6_1000_0.5_5_1/ -m out_phocnet_6_1000_0.5_5_1/model_best.pth.tar -hsg l_6_n_1000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_7_1000_0.5_5_1/ -m out_phocnet_7_1000_0.5_5_1/model_best.pth.tar -hsg l_7_n_1000_s_0.5

#histogram of all lengths,0.5 split, different num of labels
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_1000_0.5_5_1/ -m out_phocnet_all_1000_0.5_5_1/model_best.pth.tar -hsg l_all_n_1000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_2000_0.5_5_1/ -m out_phocnet_all_2000_0.5_5_1/model_best.pth.tar -hsg l_all_n_2000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.5_5_1/ -m out_phocnet_all_5000_0.5_5_1/model_best.pth.tar -hsg l_all_n_5000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_8000_0.5_5_1/ -m out_phocnet_all_8000_0.5_5_1/model_best.pth.tar -hsg l_all_n_8000_s_0.5
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_10000_0.5_5_1/ -m out_phocnet_all_10000_0.5_5_1/model_best.pth.tar -hsg l_all_n_10000_s_0.5

#histogram of all lengths,num of labels 5000, different splits
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.1_5_1/ -m out_phocnet_all_5000_0.1_5_1/model_best.pth.tar -hsg l_all_n_5000_s_0.1
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.3_5_1/ -m out_phocnet_all_5000_0.3_5_1/model_best.pth.tar -hsg l_all_n_5000_s_0.3
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.7_5_1/ -m out_phocnet_all_5000_0.7_5_1/model_best.pth.tar -hsg l_all_n_5000_s_0.7
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.9_5_1/ -m out_phocnet_all_5000_0.9_5_1/model_best.pth.tar -hsg l_all_n_5000_s_0.9

#histogram on sequestered data
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_3_500_sequestered/ -m out_phocnet_3_500_known/model_best.pth.tar -hsg l_3_n_500_sequestered
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_7_2000_sequestered/ -m out_phocnet_7_2000_known/model_best.pth.tar -hsg l_7_n_2000_sequestered
python test_phoc_net.py -d ../create_dream_dataset/out_dream_dataset_all_5000_sequestered/ -m out_phocnet_all_5000_known/model_best.pth.tar -hsg l_all_n_5000_sequestered
