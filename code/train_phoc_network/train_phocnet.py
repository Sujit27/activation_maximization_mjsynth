import sys
sys.path.append("../../")
import argparse
import logging
from pathlib import Path

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
from library.phoc_network.phoc_dataset import *
from library.phoc_network import cosine_loss
from library.phoc_network.phoc_net import *
from library.dream_reader import *


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def train():
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')
    # argument parsing
    parser = argparse.ArgumentParser()   
    # input data and output model save locations arguments
    parser.add_argument('--train_data_root', '-tr', action='store', type=str, default=None,help='Location of input dream data for training. Default: None')
    parser.add_argument('--test_data_root', '-ts', action='store', type=str, default=None,help='Location of input dream data for test. Default: None')
    parser.add_argument('--output_dir', '-od', action='store', type=str, default="out",help='Location of saving output model. Default: out')
    parser.add_argument('--save_model', '-sm', action='store', type=str, default="PhocNet.pt",help='Name of the output model. Default: PhocNet.pt')
    # - train arguments
    parser.add_argument('--num_epochs', '-ep', action='store', type=int, default=200,
                        help='Number of epochs for training. Default: 200')

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
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=64,
                        help='The batch size after which the gradient is computed. Default: 64')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.0000,
                        help='The weight decay for SGD training. Default: 0.0000')
    
    # - experiment arguments
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')

    args = parser.parse_args()

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    Path(args.output_dir).mkdir(parents=True,exist_ok=True)

    # prepare datset loader

    train_data_set = PhocDataset(args.train_data_root)
    train_loader = DataLoader(train_data_set,batch_size=args.batch_size,shuffle=True,num_workers=0)

    if args.test_data_root is not None:
        test_data_set = PhocDataset(args.test_data_root)
        test_loader = DataLoader(test_data_set,batch_size=args.batch_size,shuffle=False,num_workers=0)

    # prepare the dream reader
    lex_txt_file = "../../lexicon.txt"
    with open(lex_txt_file) as f:
        lex_list = f.readlines()
    lex_list = [word[:-1] for word in lex_list] # deleting 'n at the end of each line
    dream_reader = DreamReader(lex_list)

    # load CNN
    logger.info('Preparing PHOCNet...')

    cnn = PHOCNet(n_out=train_data_set[0][1].shape[0],input_channels=1,gpp_type='spp',pooling_levels=4)

    cnn.init_weights()

    criterion = nn.BCEWithLogitsLoss(size_average=True)

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


    for epoch in range(args.num_epochs):
        cnn.train()
        training_acc_list = []
        test_acc_list = []
        training_loss_list = []
        test_loss_list = []
        for i,data in enumerate(train_loader,0):
            imgs,embeddings,_,words = data

            imgs = imgs.to(device)
            embeddings = embeddings.to(device)

            optimizer.zero_grad()
            outputs = cnn(imgs)

            loss = criterion(outputs, embeddings) / args.batch_size
            loss.backward()
            optimizer.step()

            training_acc = evaluate_cnn(dream_reader,outputs,words)
            training_loss_list.append(loss.item())
            training_acc_list.append(training_acc)

        print("Epoch : {}, Training loss: {}, Training accuracy: {}".format(epoch,sum(training_loss_list)/len(training_loss_list),sum(training_acc_list)/len(training_acc_list)))
        #print("End of training epoch :",epoch)
        
        if args.test_data_root is not None:
            if epoch % 10 == 9:
                cnn.eval()
                for i,data in enumerate(test_loader,0):
                    imgs,embeddings,_,words = data

                    imgs = imgs.to(device)
                    embeddings = embeddings.to(device)

                    outputs = cnn(imgs)
                    test_loss = criterion(outputs, embeddings) / args.batch_size
                    test_acc = evaluate_cnn(dream_reader,outputs,words)
                    test_loss_list.append(test_loss.item())
                    test_acc_list.append(test_acc)

                print("Epoch : {}, Validation loss: {}, Validation accuracy: {}".format(epoch,sum(test_loss_list)/len(test_loss_list),sum(test_acc_list)/len(test_acc_list)))


    torch.save(cnn.state_dict(), os.path.join(args.output_dir,args.save_model))

def evaluate_cnn(dream_reader,output,words):
    indices_pred,words_pred = dream_reader.read_from_network_output(output)
    indices_target = dream_reader.convert_words_to_indices(words)

    acc_score = accuracy_score(indices_target,indices_pred)

    return acc_score

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()
