#!/bin/bash

pip3 install --user --upgrade git+https://github.com/anguelos/dagtasets

DIR_option1="~/.local/lib/python3.5/site-packages/dagtasets"
DIR_option2="~/.local/lib/python3.6/site-packages/dagtasets"
DIR_option3="~/.local/lib/python3.7/site-packages/dagtasets"
DIR_option4="~/.local/lib/python3.8/site-packages/dagtasets"


if [ -d "$DIR_option1" ]
then
    cp util.py mjsynth.py "$DIR_option1"
elif [ -d "$DIR_option2" ]
then
    cp util.py mjsynth.py "$DIR_option2"
elif [ -d "$DIR_option3" ]
then
    cp util.py mjsynth.py "$DIR_option3" 
elif [ -d "$DIR_option4" ]
then
    cp util.py mjsynth.py "$DIR_option4"  
else
    echo "dagtasets not found in ~/.local/lib/python3.#/site-packages"
fi

echo "Data will be downloaded at $1"
python3 create_dataset.py "$1"

data_path1="$1/raw/"
data_path2="$1/raw/90kDICT32px"

mv "$data_path2/*" "$data_path1"

