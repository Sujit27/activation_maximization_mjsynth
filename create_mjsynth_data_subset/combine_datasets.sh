#!/bin/bash

cd /var/tmp/on63ilaw/mjsynth/

cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_3.zip .
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_4.zip .
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_5.zip .
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_6.zip .
cp /proj/cipdata/on63ilaw/mjsynth/raw_files/raw_7.zip .

mkdir raw_tmp

unzip -q -o raw_3.zip
mv raw raw3
mv raw3 raw_tmp/

unzip -q -o raw_4.zip
mv raw raw4
mv raw4 raw_tmp/

unzip -q -o raw_5.zip
mv raw raw5
mv raw5 raw_tmp/

unzip -q -o raw_6.zip
mv raw raw6
mv raw6 raw_tmp/

unzip -q -o raw_7.zip
mv raw raw7
mv raw7 raw_tmp/


rm -r raw/
rm *.csv *.zip


