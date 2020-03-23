import argparse
import logging

import numpy as np
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import tqdm

import copy
from phoc_dataset import *
from cosine_loss import *
from phoc_net import *
from torch.utils.data.dataloader import DataLoaderIter


def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def train():
    logger = logging.getLogger('PHOCNet-Experiment::train')
    logger.info('--- Running PHOCNet Training ---')
    # argument parsing
    parser = argparse.ArgumentParser()    
    # - train arguments
    parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser, default='60000:1e-4,100000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'70000:1e-4,80000:1e-5\' means learning rate 1e-4 up to step 70000 and 1e-5 till 80000.')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--delta', action='store', type=float, default=1e-8,
                        help='Epsilon if solver is Adam. Default: 1e-8')
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 100')
    parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 500')
    parser.add_argument('--iter_size', '-is', action='store', type=int, default=10,
                        help='The batch size after which the gradient is computed. Default: 10')
    parser.add_argument('--batch_size', '-bs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    
    # - experiment arguments
    parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    
    
    args = parser.parse_args()

    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).iteritems():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # prepare datset loader

    train_data_set = PhocDataset(root_dir='../../dataset/')

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

    train_loader_iter = DataLoaderIter(loader=train_loader)

    # load CNN
    logger.info('Preparing PHOCNet...')

    cnn = PHOCNet(n_out=train_data_set[0][1].shape[0],input_channels=1,gpp_type='spp',pooling_levels=([1], [5]))

    cnn.init_weights()


    loss_selection = 'BCE' # or 'cosine'
    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        loss = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('not supported loss function')

    # move CNN to GPU
    cnn.to(device)

    # run training
    lr_cnt = 0
    max_iters = args.learning_rate_step[-1][0]
    if args.solver_type == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(), args.learning_rate_step[0][1],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.solver_type == 'Adam':
        optimizer = torch.optim.Adam(cnn.parameters(), args.learning_rate_step[0][1],
                                    weight_decay=args.weight_decay)


    optimizer.zero_grad()
    logger.info('Training:')
    for iter_idx in range(max_iters):
        if iter_idx % args.test_interval == 0: # and iter_idx > 0:
            logger.info('Evaluating net after %d iterations', iter_idx)
            evaluate_cnn(cnn=cnn,
                         dataset_loader=val_loader,
                         args=args) 
####################### START Cleaning here                         
        for _ in range(args.iter_size):
            if train_loader_iter.batches_outstanding == 0:
                train_loader_iter = DataLoaderIter(loader=train_loader)
                logger.info('Resetting data loader')
            word_img, embedding, _, _ = train_loader_iter.next()
            if args.gpu_id is not None:
                if len(args.gpu_id) > 1:
                    word_img = word_img.cuda()
                    embedding = embedding.cuda()
                else:
                    word_img = word_img.cuda(args.gpu_id[0])
                    embedding = embedding.cuda(args.gpu_id[0])

            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)
            output = cnn(word_img)
            ''' BCEloss ??? '''
            loss_val = loss(output, embedding)*args.batch_size
            loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # mean runing errors??
        if (iter_idx+1) % args.display == 0:
            logger.info('Iteration %*d: %f', len(str(max_iters)), iter_idx+1, loss_val.data[0])

        # change lr
        if (iter_idx + 1) == args.learning_rate_step[lr_cnt][0] and (iter_idx+1) != max_iters:
            lr_cnt += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate_step[lr_cnt][1]

        #if (iter_idx + 1) % 10000 == 0:
        #    torch.save(cnn.state_dict(), 'PHOCNet.pt')
            # .. to load your previously training model:
            #cnn.load_state_dict(torch.load('PHOCNet.pt'))

    #torch.save(cnn.state_dict(), 'PHOCNet.pt')
    my_torch_save(cnn, 'PHOCNet.pt')


def evaluate_cnn(cnn, dataset_loader, args):
    logger = logging.getLogger('PHOCNet-Experiment::test')
    # set the CNN in eval mode
    cnn.eval()
    logger.info('Computing net output:')
    qry_ids = [] #np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm.tqdm(dataset_loader)):
        if args.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(args.gpu_id[0])
            embedding = embedding.cuda(args.gpu_id[0])
            #word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)
        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        ''' BCEloss ??? '''
        output = torch.sigmoid(cnn(word_img))
        #output = cnn(word_img)
        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0,0]
        if is_query[0] == 1:
            qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]

    '''
    # find queries
    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]
    # remove stopwords if needed
    
    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    ave_precs_qbe = map_from_query_test_feature_matrices(query_features = qry_outputs,
                                                         test_features=outputs,
                                                         query_labels = qry_class_ids,
                                                         test_labels=class_ids,
                                                         metric='cosine',
                                                         drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[ave_precs_qbe > 0])*100)



    # clean up -> set CNN in train mode again
    cnn.train()

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()