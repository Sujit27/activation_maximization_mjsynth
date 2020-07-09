import os
import json
import random
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_paths', nargs='+',required=True,help='list of file path where json files are available')
parser.add_argument('-n',type=int,default = 4, dest='num_words',help='Number of common words to generate')

cmd_args = parser.parse_args()

def main():
    label_dicts = []
    for model_path in cmd_args.model_paths:
        dict_file = os.path.join(model_path,'label_dict.json')
        with open(dict_file) as json_file:
            label_dict = json.load(json_file)
        label_dicts.append(label_dict)

    word_lists = []
    for label_dict in label_dicts:
        words = [word for word in label_dict.values()]
        word_lists.append(set(words))

    common_words = set.intersection(*word_lists)
    random.seed(0)
    chosen_words = random.sample(common_words,cmd_args.num_words)
    for label_dict in label_dicts:
        label_dict = {word:index for index,word in label_dict.items()}
        for word in chosen_words:
            print('{} : {}'.format(word,label_dict[word]))

if __name__ == "__main__":
    main()

