# coding: utf-8
import sys
sys.path.append("../../library/dict_network/")
import json
import pandas as pd
import torch.nn.functional as F
from dict_net import *
from dict_net_dataset import *

root_dir = "/var/tmp/on63ilaw/mjsynth/synth_images/"
json_file = "out_all_5000_sequestered/label_dict.json"
saved_model = "out_all_5000_sequestered/model_best.pth.tar"
output_file = "rendered_image_preds_sequestered"
def main():
    ds = DictNetDataset(root_dir)
    net = DictNet(5000)
    net.load_state_dict(torch.load(saved_model)['state_dict'])
    net = nn.DataParallel(net)
    net.eval()
    dataloader = torch.utils.data.DataLoader(ds,batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    with open(json_file) as f:
        label_dict = json.load(f)
    label_dict = {int(key):val for key,val in label_dict.items()} 

    prediction_dict = {}
    
    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            images,targets = data
            images = images.to(device)
            outputs = F.softmax(net(images),dim=1)
            value,index = torch.max(outputs,1)
            prediction_dict[i] = (index.item(),value.item(),ds.words[i])
            if i%5000==4999:
                print(i) 
        
    df = pd.DataFrame.from_dict(prediction_dict,orient='index',columns=['predicted_label','confidence','word'])
    df = df.sort_values(by=['predicted_label','confidence'],ascending=[True,False])
    df.to_csv(output_file,index=False)

if __name__ == "__main__":
    main()
