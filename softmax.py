import numpy as np
import json
word2class={int(k):v for k,v in json.load(open("code/train_dict_network/out_all_5000_sequestered/label_dict.json")).items()}


lines=[l.split(",") for l in open("code/train_dict_network/rendered_image_preds.csv").read().strip().split("\n")[1:]]

probs=np.array([float(l[1]) for l in lines])
labels=np.array([int(l[0]) for l in lines])
captions=np.array([l[2] for l in lines])

keep=np.random.rand(len(labels))>.5



correct=[word2class[labels[keep][n]]==labels[keep][n] for n in range(sum(keep))]
labels[keep]
word2class={int(k):v for k,v in json.load(open("code/train_dict_network/out_all_5000_sequestered/label_dict.json")).items()}

k_probs=probs[keep]
k_captions=captions[keep]
k_labels=labels[keep]

correct=[word2class[k_labels[n]]==k_captions[n] for n in range(sum(keep)) if k_probs[n]>.9999];print((len(correct),sum(correct)/len(correct)))
correct=[word2class[k_labels[n]]==k_captions[n] for n in range(sum(keep)) if k_probs[n]>.99999];print((len(correct),sum(correct)/len(correct)))
correct=[word2class[k_labels[n]]==k_captions[n] for n in range(sum(keep)) if k_probs[n]>.999999];print((len(correct),sum(correct)/len(correct)))
correct=[word2class[k_labels[n]]==k_captions[n] for n in range(sum(keep)) if k_probs[n]>.9999999];print((len(correct),sum(correct)/len(correct)))

