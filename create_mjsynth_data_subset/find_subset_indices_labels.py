import torch
import os
import dagtasets.mjsynth as mj
import re

def find_indices_labels(ds,lex):

    file_names = [item for item in ds.filenames]
    file_names = [os.path.basename(item) for item in file_names]
    file_names = [os.path.splitext(item)[0] for item in file_names]
    words = [" ".join(re.findall("[a-zA-Z]+",item)) for item in file_names]
    words = [item.lower() for item in words]
    targets = [int((item.split("_"))[0]) for item in file_names]

    indices = []
    index = 0
    for word in words:
        for item in lex:
            if word == item:
                indices.append(index)
        index += 1
#        if index == 100:
#            break

    return indices

def main():
    ds = mj.MjSynthWS('/var/tmp/on63ilaw/mjsynth')
    indices,labels = find_indices_labels(ds,'lex_a_len9.txt')

if __name__ == "__main__":
    main()

