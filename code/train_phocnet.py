import sys
sys.path.append("../library")
sys.path.append("../library/phoc_net")
import argparse
import logging

import numpy as np
from sklearn.metrics import accuracy_score
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data  import SubsetRandomSampler
import tqdm

import copy
from phoc_dataset import *
from cosine_loss import *
from phoc_net import *
from retrieval import *


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def train():
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')
    # argument parsing
    parser = argparse.ArgumentParser()    
    # - train arguments
    parser.add_argument('--num_epochs', '-ep', action='store', type=int, default=50,
                        help='Number of epochs for training. Default: 50')

    parser.add_argument('--learning_rate', '-lr',action='store',type=float, default=0.0001,
            help='Learning rate for training. Default: 0.0001')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=50,
                        help='The number of batches after which to display the loss values. Default: 50')
    parser.add_argument('--test_interval', action='store', type=int, default=500,
                        help='The number of batches after which to evaluate the PHOCNet. Default: 500')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=128,
                        help='The batch size after which the gradient is computed. Default: 128')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.0000,
                        help='The weight decay for SGD training. Default: 0.0000')
    
    # - experiment arguments
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    parser.add_argument('--num_word_labels', '-nwl', action='store', type=int, default=None,
                        help='The number of word labels. None means all word labels considered for training. Default: None')

    
    args = parser.parse_args()


    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # prepare datset loader

    train_data_set = PhocDataset('/var/tmp/on63ilaw/mjsynth',args.num_word_labels)

    # split training and validation data
    validation_split = 0.1
    dataset_size = len(train_data_set)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    shuffle_dataset = True
    #random_seed= np.random.randint(100)
    random_seed = 0

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)


    train_loader = DataLoader(train_data_set,batch_size=args.batch_size,sampler=train_sampler,num_workers=8)
    val_loader = DataLoader(train_data_set,batch_size=args.batch_size,sampler=valid_sampler,num_workers=8)


    # load CNN
    logger.info('Preparing PHOCNet...')

    cnn = PHOCNet(n_out=train_data_set[0][1].shape[0],input_channels=1,gpp_type='spp',pooling_levels=4)

    cnn.init_weights()


    loss_selection = 'BCE' #or 'cosine' 
    if loss_selection == 'BCE':
        criterion = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        criterion = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('not supported loss function')

    # move CNN to GPU
    cnn.to(device)

    # run training
    if args.solver_type == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.solver_type == 'Adam':
        optimizer = torch.optim.Adam(cnn.parameters(), args.learning_rate,
                                    weight_decay=args.weight_decay)


    logger.info('Training:')
    for epoch in range(args.num_epochs):
        cnn.train()
        training_acc_list = []
        validation_acc_list = []
        for i,data in enumerate(train_loader,0):
            imgs,embeddings,class_ids = data

            imgs = imgs.to(device)
            embeddings = embeddings.to(device)
            class_ids = class_ids.to(device)

            optimizer.zero_grad()
            outputs = cnn(imgs)

            loss = criterion(outputs, embeddings) * args.batch_size
            loss.backward()
            optimizer.step()

            if i % args.display == 0:
                print("Epoch: {}, Batch : {}, Loss : {}".format(epoch,i,loss.item()))
            if i % args.test_interval == 0:
                training_acc = evaluate_cnn(train_data_set,class_ids,outputs)
                print("Traning accuracy : {}".format(training_acc))
                training_acc_list.append(training_acc)
                #evaluate_cnn(train_data_set,class_ids,embeddings) # to check that evaluate_cnn work

        cnn.eval()
        for i,data in enumerate(val_loader,0):
            if i % 20 == 0:
                imgs,embeddings,class_ids = data

                imgs = imgs.to(device)
                embeddings = embeddings.to(device)
                class_ids = class_ids.to(device)

                outputs = cnn(imgs)
                validation_acc = evaluate_cnn(train_data_set,class_ids,outputs)
                validation_acc_list.append(validation_acc)

        print("End of epoch : {}, Training accuracy : {}, Validation accuracy : {}".format(epoch,sum(training_acc_list)/len(training_acc_list),sum(validation_acc_list)/len(validation_acc_list)))



    torch.save(cnn.state_dict(), '../models/PHOCNet.pt')


def evaluate_cnn(dataset, class_ids, outputs):
    class_ids = class_ids.cpu()
    outputs = outputs.cpu().detach().numpy()
    output_similarity = np.dot(dataset.word_string_embeddings,np.transpose(outputs))
    indices_predicted = np.argmax(output_similarity,0)
    class_ids_predicted = []
    for index in indices_predicted:
        _,_,class_id_predicted = dataset[index]
        class_ids_predicted.append(class_id_predicted)

    class_ids_predicted = np.array(class_ids_predicted).flatten()
    acc_score = accuracy_score(class_ids,class_ids_predicted)
    #print("Training accuracy score :",acc_score)
    return acc_score

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()
