import dagtasets.mjsynth as mj 
import os
import shutil
import sys
import argparse
from tqdm import trange
from find_subset_indices_labels import *

## creates subset of the mjsynth dataset according to the word length provided and saves the output in a directory raw2 at the root directory
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_root', '-r',action='store',type=str, default='/var/tmp/on63ilaw/mjsynth',help='Location where mjsynth dataset is present. The new subset dataset will also be created in a new directory raw2 at this location')
parser.add_argument('--word_len', '-wl',action='store',type=int, default=8,help='Length of word for making a subset of the original Mjsynth dataset present at the data root location')

args = parser.parse_args()

def subset_dataset(root,word_len):
    lexicon_file = os.path.join(root,'raw','lexicon.txt')
    words_list = sorted([w for w in open(lexicon_file).read().strip().split("\n") if (len(w)==word_len)])
    
    print("Number of word labels in the subset dataset", len(words_list))
       
    try:
        os.mkdir(os.path.join(root,'raw2'))
    except:
        print("raw2 file already exists at destination. Remove or rename it")
        return
    
    ds = mj.MjSynthWS(root)
    print("Finding indices...")
    indices = find_indices_labels(ds,words_list)

    dst = os.path.join(root,'raw2')
    print("Copying files from root...")
    
    for i in trange(len(indices)):
        src = os.path.join(root,'raw',ds.filenames[indices[i]])
        shutil.copy(src,dst)
    
    print("Copy complete. Writing annotation text file...")
    filenames = [ds.filenames[indices[i]] for i in range(len(indices))]
    filename_list = [os.path.split(filename)[1] for filename in filenames]
    
    with open(os.path.join(root,'raw2','annotation_train.txt'),'w') as f:
        for item in filename_list:
            f.write('%s\n' % item)
    


def main():
    subset_dataset(args.data_root,args.word_len)

if __name__ == "__main__":
    main()
