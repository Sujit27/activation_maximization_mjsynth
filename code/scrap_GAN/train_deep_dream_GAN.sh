#i /bin/bash

# bash script to run train simple GAN type dreamer with multiple hyperparameter settings
declare -i batch_size
lr=0.001
batch_size=8
python_file="train_deep_dream_GAN.py" 
echo "(($batch_size)+2)"
#python3 "$python_file" -lr "$lr" 

