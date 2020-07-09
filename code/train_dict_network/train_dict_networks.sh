#!/bin/bash

cd /var/tmp/on63ilaw/mjsynth
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_3_partitions_500.zip .
unzip -q -o raw_3_partitions_500.zip

cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
python train_dict_net.py  -o out_3_500_known -d /var/tmp/on63ilaw/mjsynth/known -ne 10 > out_3_500_known.txt
python train_dict_net.py  -o out_3_500_sequestered -d /var/tmp/on63ilaw/mjsynth/sequestered -ne 10 > out_3_500_sequestered.txt


cd /var/tmp/on63ilaw/mjsynth
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_7_partitions_2000.zip .
unzip -q -o raw_7_partitions_2000.zip

cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
python train_dict_net.py  -o out_7_2000_known -d /var/tmp/on63ilaw/mjsynth/known -ne 8 > out_7_2000_known.txt
python train_dict_net.py  -o out_7_2000_sequestered -d /var/tmp/on63ilaw/mjsynth/sequestered -ne 8 > out_7_2000_sequestered.txt

cd /var/tmp/on63ilaw/mjsynth
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_all_partitions_5000.zip .
unzip -q -o raw_all_partitions_5000.zip

cd /proj/cipdata/on63ilaw/mjsynth/code/train_dict_network
python train_dict_net.py  -o out_all_5000_known -d /var/tmp/on63ilaw/mjsynth/known -ne 8 > out_all_5000_known.txt
python train_dict_net.py  -o out_all_5000_sequestered -d /var/tmp/on63ilaw/mjsynth/sequestered -ne 8 > out_all_5000_sequestered.txt


