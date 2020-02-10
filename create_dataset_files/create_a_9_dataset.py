import dagtasets.mjsynth as mj 
import os
import shutil
import sys
from find_subset_indices_labels import *
script_path = os.getcwd()
sys.path.append(script_path)

def create_a_9_dataset(root):
	os.chdir(root)
	os.mkdir('raw2')
	
	ds = mj.MjSynthWS('.') 
	indices,_ = find_indices_labels(ds,os.path.join(script_path,"lex_a_len9.txt")
	
	dst = os.path('./raw2/')
	
	for i in range(len(indices)):
		src = os.path.join('raw/',ds.filenames[indices[i]])
		shutil.copy(src,dst)
		
	filenames = [ds.filenames[indices[i]] for i in range(len(indices))]
	filename_list = [os.path.split(filename)[1] for filename in filenames]
	
	with open('raw2/annotation_train.txt','w') as f:
		for item in filename_list:
			f.write('%s\n' % item)
	


def main():
    #root = '/mnt/c/Users/User/Desktop/mjsynth'
    if len(sys.argv) < 2:
		print("Pass the location of the data as a parameter to the file. For example /var/tmp/")
		return 1
	else:
		root = sys.argv[1]
		create_a_9_dataset(root)
