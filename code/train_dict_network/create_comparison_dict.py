# coding: utf-8
import sys
sys.path.append("../../library/dict_network/")
import json
import pandas as pd
import torch.nn.functional as F
from dict_net import *
from dict_net_dataset import *
import editdistance as ed

predicted_csv_file = "rendered_image_preds.csv"
json_file = "out_all_5000_known/label_dict.json"

def main():
    with open(json_file) as f:
        label_dict = json.load(f)
    label_dict = {int(key):val for key,val in label_dict.items()}
    actual_df = pd.DataFrame.from_dict(label_dict,orient='index')
    actual_df.reset_index(level=0,inplace=True)
    actual_df.columns = ['label','actual_word']

    predicted_df = pd.read_csv(predicted_csv_file)
    predicted_df = predicted_df.loc[predicted_df.groupby('predicted_label')['confidence'].idxmax()]
    predicted_df = predicted_df.rename(columns={'word':'predicted_word','predicted_label':'label'})

    combined_df = actual_df.merge(predicted_df,on='label')
    combined_df = combined_df.sort_values(by=['label'])
    combined_df.actual_word

    combined_df['edit_distance'] = combined_df.apply(lambda row: ed.distance(row.actual_word,row.predicted_word),axis=1)

    print("Number of total rows :",combined_df.shape[0])
    print("Number of rows where editdistance is zero:",(combined_df.edit_distance == 0).count())

    combined_df.to_csv("comparison.csv")


if __name__ == "__main__":
    main()
