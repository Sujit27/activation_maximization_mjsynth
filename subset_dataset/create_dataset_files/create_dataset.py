## creates the mjsynth dataset the location provided.
## usage: $python create_dataset.py $download_destination

import dagtasets
import os
import shutil
import sys

url = 'http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz'

def create_dataset(root):
    
    # download the tar
    zip_folder = os.path.join(root,'zips')
    os.mkdir(zip_folder)
    dagtasets.util.resumable_download(url,zip_folder)
    
    # extract the tar
    archive_path = os.path.join(zip_folder,"mjsynth.tar.gz")
    tmp_folder = os.path.join(root,'tmp')
    mj_subdirs = ["mnt", "ramdisk", "max", "90kDICT32px"]
    mjsynth_root = os.path.join(tmp_folder, *mj_subdirs)
    dagtasets.util.extract(archive_path,tmp_folder)
    
    # move the extracted images
    dest_dir = os.path.join(root,'raw')
    os.mkdir(dest_dir)
    shutil.move(mjsynth_root,dest_dir)
    
def main():
    #root = '/mnt/c/Users/User/Desktop/mjsynth'
    root = sys.argv[1]
    create_dataset(root)
    
if __name__ == "__main__":
    main()
    

