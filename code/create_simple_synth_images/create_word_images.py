import sys
sys.path.append("../../library")

import os
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from helper_functions import *

frequent_word_list_file = "lexicon.txt"
font_file = "../../font/NotoSans-Regular.ttf"
image_save_dir = '/var/tmp/on63ilaw/mjsynth/synth_images/raw'


def main():
    with open(frequent_word_list_file) as f:
        words = [line.rstrip() for line in f]
    
    Path(image_save_dir).mkdir(parents=True,exist_ok=True)
    W,H = (128,32)
    font = ImageFont.truetype(font_file, 16)
    for i,word in enumerate(words):
        im = Image.new("RGBA", (W,H), color='white')
        draw = ImageDraw.Draw(im)
        w, h = draw.textsize(word,font=font)
        draw.text(((W-w)/2,(H-h)/2), word, font=font,fill='black')
        image_file_name = "pil_" + word + "_" + str(i) + ".jpg"
        im.save(os.path.join(image_save_dir,image_file_name))

        if i == 100:
            break:

    create_annotation_txt(image_save_dir)

if __name__ == "__main__":
    main()
