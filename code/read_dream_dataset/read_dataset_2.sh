#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_7_2000_0.1_5_1 -o out_phocnet_7_2000_0.1_5_1   -ne 50 > out_train_result_7_2000_0.1_5_1.txt
#
#rm out_phocnet_7_2000_0.1_5_1/checkpoint.pth.tar
#
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_7_5000_0.1_5_1 -o out_phocnet_7_5000_0.1_5_1   -ne 50 > out_train_result_7_5000_0.1_5_1.txt
#
#rm out_phocnet_7_5000_0.1_5_1/checkpoint.pth.tar
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_1000_0.1_5_1 -o out_phocnet_all_1000_0.1_5_1   -ne 50 > out_train_result_all_1000_0.1_5_1.txt
#
#rm out_phocnet_all_1000_0.1_5_1/checkpoint.pth.tar
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_1000_0.5_5_1 -o out_phocnet_all_1000_0.5_5_1   -ne 50 > out_train_result_all_1000_0.5_5_1.txt
#
#rm out_phocnet_all_1000_0.5_5_1/checkpoint.pth.tar
#
#python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_2000_0.1_5_1 -o out_phocnet_all_2000_0.1_5_1   -ne 50 > out_train_result_all_2000_0.1_5_1.txt
#
#rm out_phocnet_all_2000_0.1_5_1/checkpoint.pth.tar


python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_5000_0.1_5_1 -o out_phocnet_all_5000_0.1_5_1   -ne 50 > out_train_result_all_5000_0.1_5_1.txt

rm out_phocnet_all_5000_0.1_5_1/checkpoint.pth.tar


python train_phocnet_on_dream_dataset.py -d ../create_dream_dataset/out_dream_dataset_all_10000_0.1_5_1 -o out_phocnet_all_10000_0.1_5_1   -ne 50 > out_train_result_all_10000_0.1_5_1.txt

rm out_phocnet_all_10000_0.1_5_1/checkpoint.pth.tar

