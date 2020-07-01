import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_annotation_txt(files_path):
    '''
    creates a txt file listing the names of all the jpg files
    in the given directory
    '''
    file_list = glob.glob(os.path.join(files_path,"*.jpg"))
    output_file_name = os.path.join(files_path,"annotation_train.txt")

    print("Creating annotation txt file")

    with open(output_file_name,'w') as f:
        for item in file_list:
            f.write("%s\n" % os.path.basename(item))
 
def main():
    frequent_word_list_file = "most_freq_words_20000.txt"
    with open(frequent_word_list_file) as f:
        words = [line.rstrip() for line in f]
    
    font_file = "../../font/NotoSans-Regular.ttf"
    image_save_dir = '/var/tmp/on63ilaw/mjsynth/synth_images/raw'
    Path(image_save_dir).mkdir(parents=True,exist_ok=True)

    font = ImageFont.truetype(font_file, 16)
    for i,word in enumerate(words):
        im = Image.new("RGB", (128,32), color='white')
        draw = ImageDraw.Draw(im)
        draw.text((10,10), word, font=font,fill=(0,0,0))
        image_file_name = "pil_" + word + "_regular.jpg"
        im.save(os.path.join(image_save_dir,image_file_name))

        if i % 1000 == 0:
            print("{} images created".format(i))

    create_annotation_txt(image_save_dir)

if __name__ == "__main__":
    main()
