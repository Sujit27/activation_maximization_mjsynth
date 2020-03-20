#i /bin/bash

# bash script to run train simple GAN type dreamer with multiple hyperparameter settings
#declare -i batch_size
lrs=(0.0005 0.001 0.005 0.01 0.05)
batch_sizes=(16 32 64)
python_file="train_deep_dream_GAN.py" 

log_start="log"
log_end=".csv"

for lr in ${lrs[@]};do
    for batch_size in ${batch_sizes[@]};do
        log_file_name="${log_start}_${lr}_${batch_size}${log_end}"
        python3 "$python_file" -lr "$lr" -bs $batch_size > "$log_file_name"
    done
done

