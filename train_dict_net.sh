#!/bin/bash

cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network

#python train_dict_net.py -n 1000 -o out9 -ne 20
#python train_dict_net.py -n 2000 -o out9 -ne 20
python train_dict_net.py -n 5000 -o out7 -ne 10
python train_dict_net.py -n 10000 -o out7 -ne 10


cd /var/tmp/on63ilaw/mjsynth/
rm -r ./*
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_8.zip .
unzip -q raw_8.zip

cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
python train_dict_net.py -n 5000 -o out8 -ne 10
