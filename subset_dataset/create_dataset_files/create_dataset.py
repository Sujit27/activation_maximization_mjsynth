
import dagtasets
import os
import shutil
import sys
import argparse

## creates the mjsynth dataset at the location provided.


url = 'http://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_root', '-r',action='store',type=str, default='/var/tmp/on63ilaw/mjsynth',help='Location where mjsynth dataset will be stored')

def create_dataset(root):
    # make root directory if does not exist already
    os.makedirs(root,exist_ok=True)
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

    print("Files extracted")
    
    # move the extracted images
    dest_dir = os.path.join(root,'raw')
    os.mkdir(dest_dir)
    shutil.move(mjsynth_root,dest_dir)

    for filename in os.listdir(os.path.join(dest_dir,'90kDICT32px')):
        shutil.move(os.path.join(dest_dir,'90kDICT32px',filename),os.path.join(dest_dir,filename))
    
    print("Files moved")
def main():
    args = parser.parse_args()
    create_dataset(args.data_root)
    
if __name__ == "__main__":
    main()
    

