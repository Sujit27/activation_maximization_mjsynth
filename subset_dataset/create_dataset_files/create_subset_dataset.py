## creates subset of the mjsynth dataset according to the words in the .txt file provideda## see a sample txt file in this directory
## usage: $python create_subset_dataset.py <mjsynth_data_loccation> <txt file> 
import dagtasets.mjsynth as mj 
import os
import shutil
import sys
from find_subset_indices_labels import *
script_path = os.getcwd()
sys.path.append(script_path)

def subset_dataset(root,lex_txt):
    #os.chdir(root)
    os.mkdir(os.path.join(root,'raw2'))
    #print('new directory created at {}'.format(root))

    ds = mj.MjSynthWS(root)
    indices,_ = find_indices_labels(ds,os.path.join(script_path,lex_txt))
    
    dst = os.path.join(root,'raw2')
    
    for i in range(len(indices)):
        src = os.path.join(root,'raw',ds.filenames[indices[i]])
        shutil.copy(src,dst)
        
    filenames = [ds.filenames[indices[i]] for i in range(len(indices))]
    filename_list = [os.path.split(filename)[1] for filename in filenames]
    
    with open(os.path.join(root,'raw2','annotation_train.txt'),'w') as f:
        for item in filename_list:
            f.write('%s\n' % item)
    


def main():
    #root = '/mnt/c/Users/User/Desktop/mjsynth'
    if len(sys.argv) < 3:
        print("Pass the location of the data as a parameter and txt file as command line parameters. For example python create_subset_dataset.py /var/tmp/ lex_###.txt")
        return 1
    else:
        root = sys.argv[1]
        lex_txt = sys.argv[2] # text file with one word per each line
        subset_dataset(root,lex_txt)

if __name__ == "__main__":
    main()
