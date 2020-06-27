#!/bin/bash
#cd /var/tmp/on63ilaw/mjsynth
#
#cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_3.zip .
#unzip -q -o raw_3.zip
#cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
#python train_dict_net_new.py -n 1000 -o out_3_1000 -d /var/tmp/on63ilaw/mjsynth -ne 10 > out_3_1000.txt


cd /var/tmp/on63ilaw/mjsynth
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_4.zip .
unzip -q -o raw_4.zip
cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
python train_dict_net_new.py -n 1000 -o out_4_1000 -d /var/tmp/on63ilaw/mjsynth -ne 10 > out_4_1000.txt


cd /var/tmp/on63ilaw/mjsynth
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_5.zip .
unzip -q -o raw_5.zip
cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
python train_dict_net_new.py -n 1000 -o out_5_1000 -d /var/tmp/on63ilaw/mjsynth -ne 10 > out_5_1000.txt
