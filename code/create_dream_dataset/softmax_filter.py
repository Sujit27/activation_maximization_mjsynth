import numpy as np
import json

def get_softmax_correct_labels(softmax_predictions_csv,label_dict_json,threshold=0.99999):

    word2class={int(k):v for k,v in json.load(open(label_dict_json)).items()}


    lines=[l.split(",") for l in open(softmax_predictions_csv).read().strip().split("\n")[1:]]

    probs=np.array([float(l[1]) for l in lines])
    labels=np.array([int(l[0]) for l in lines])
    captions=np.array([l[2] for l in lines])

    keep=np.random.rand(len(labels))>.5

    #correct=[word2class[labels[keep][n]]==labels[keep][n] for n in range(sum(keep))]
    #labels[keep]
    #word2class={int(k):v for k,v in json.load(open("./label_dict.json")).items()}

    k_probs=probs[keep]
    k_captions=captions[keep]
    k_labels=labels[keep]

    labels_guessed = {}
    correct_labels_guessed = {}

    for n in range(sum(keep)):
        if (k_probs[n]>threshold):
            labels_guessed[k_labels[n]] = k_captions[n]
            if (word2class[k_labels[n]]==k_captions[n]):
                correct_labels_guessed[k_labels[n]] = k_captions[n]
    
    return labels_guessed,correct_labels_guessed

def main():
    softmax_predictions_csv = "../train_dict_network/rendered_image_preds_known.csv"   
    label_dict_json = "../train_dict_network/out_all_5000_known/label_dict.json"
    labels_guessed_1,correct_labels_guessed_1 = get_softmax_correct_labels(softmax_predictions_csv,label_dict_json,0.9999999)
    labels_guessed_2,correct_labels_guessed_2 = get_softmax_correct_labels(softmax_predictions_csv,label_dict_json,0.999999)
    labels_guessed_3,correct_labels_guessed_3 = get_softmax_correct_labels(softmax_predictions_csv,label_dict_json,0.99999)
    labels_guessed_4,correct_labels_guessed_4 = get_softmax_correct_labels(softmax_predictions_csv,label_dict_json,0.9999)
    print(len(labels_guessed_1),len(correct_labels_guessed_1))
    print(len(labels_guessed_2),len(correct_labels_guessed_2))
    print(len(labels_guessed_3),len(correct_labels_guessed_3))
    print(len(labels_guessed_4),len(correct_labels_guessed_4))

if __name__ == "__main__":
    main()
