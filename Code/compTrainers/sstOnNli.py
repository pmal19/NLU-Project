import time
import sys
import os
import re
import torch
from torch import np
#import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
#from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
# from itertools import accumulate
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from sstEncoder import sstNet 
from Models.blocks import *

from data.dataparser import *
from data.batcher import *
from readEmbeddings import *
from datasets import *
from save import *
from TaskEncoder import *

# import Taskselector as task_selector

import pdb

class CompNetSSTonNLI(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, path1, training = True):
        super(CompNetSSTonNLI, self).__init__()
        self.encoder1 = TaskEncoder(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, False, path1, "", "", "sst")
        self.compMLP = MLP(model_dim*2, model_dim, 3, 2, True, dropout, training)

    def forward(self, s1, s2):
        a1 = self.encoder1(s1)
        a2 = self.encoder1(s1)
        # pdb.set_trace()
        features = torch.cat((a1, a2), 1)
        output = self.compMLP(features)
        return output


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
    print("Epoch start - ",epoch)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #pdb.set_trace()
        s1, s2 = data
        batchSize, _ = s1.shape
        s1 = s1.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        s2 = s2.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        if(use_cuda):
            s1, s2, target = Variable(s1.cuda()), Variable(s2.cuda()), Variable(target.cuda())
        else:
            s1, s2, target = Variable(s1), Variable(s2), Variable(target)
        optimizer.zero_grad()
        output = model(s1, s2)
        # pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # print("loss backward")
        if batch_idx == break_val:
            return
        if batch_idx % 100 == 0:
            dev_loss = 0
            for idx, (dev_data, dev_target) in enumerate(devLoader):
                sd1, sd2 = dev_data
                # pdb.set_trace()
                devbatchSize, _ = sd1.shape
                sd1 = sd1.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
                sd2 = sd2.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
                if(use_cuda):
                    sd1, sd2, dev_target = Variable(sd1.cuda()), Variable(sd2.cuda()), Variable(dev_target.cuda())
                else:
                    sd1, sd2, dev_target = Variable(sd1), Variable(sd2), Variable(dev_target)
                dev_output = model(sd1, sd2)
                dev_loss = criterion(dev_output, dev_target)
                break
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDev: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0], dev_loss.data[0]))
            save(model, optimizer, loss, 'compTrainerSSTonNLI.pth', dev_loss)


def train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
    for epoch in range(numEpochs):
        trainEpoch(epoch,20000000,trainLoader,model,optimizer,criterion,inp_dim,batchSize, use_cuda, devLoader, devbatchSize)


def main():

    local = False

    if(local):
        quoraPathTrain = '../../data/questionsTrain.csv'
        quoraPathDev = '../../data/questionsDev.csv'
        nliPathTrain = "../../../Data/snli_1.0/snliSmallaa"
        nliPathDev = "../../../Data/snli_1.0/snliSmallDevaa"
        sstPathTrain = "../../trees/train.txt"
        sstPathDev = "../../trees/dev.txt"
        glovePath = '../../../glove.6B/glove.6B.300d.txt'
    else:
        quoraPathTrain = '../../data/questionsTrain.csv'
        quoraPathDev = '../../data/questionsDev.csv'
        nliPathTrain = "/scratch/pm2758/nlu/snli_1.0/snliSmallaa"
        nliPathDev = "/scratch/pm2758/nlu/snli_1.0/snliSmallDevaa"
        sstPathTrain = "/scratch/pm2758/nlu/trees/train.txt"
        sstPathDev = "/scratch/pm2758/nlu/trees/dev.txt"
        glovePath = '/scratch/pm2758/nlu/glove.840B.300d.txt'

    use_cuda = torch.cuda.is_available()
    if(use_cuda):
        the_gpu.gpu = 0

    batchSize = 64
    learningRate = 0.001
    momentum = 0.9
    numWorkers = 5
    numEpochs = 10
    inp_dim = 300
    model_dim = 300
    num_layers = 1
    reverse = False
    bidirectional = True
    dropout = 0.1
    mlp_input_dim = 600
    mlp_dim = 300
    num_classes = 2
    num_mlp_layers = 2
    mlp_ln = True
    classifier_dropout_rate = 0.1
    training = True


    t1 = time.time()
    trainingDataset = nliDataset(nliPathTrain, glovePath)
    devDataset = nliDataset(nliPathDev, glovePath)
    print('Dataset loading time taken - ',time.time()-t1, len(devDataset))

    trainLoader = DataLoader(trainingDataset, batchSize, num_workers = numWorkers)
    devLoader = DataLoader(devDataset, len(devDataset)/1000, num_workers = numWorkers)

    sst_path = "../sstTrained"
    nli_path = "../snliTrained"
    quora_path = "../quoraTrained"

    model = CompNetSSTonNLI(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, sst_path, training)


    if(use_cuda):
        model.cuda()
    if(use_cuda):
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 1e-5)

    train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, len(devDataset))

if __name__ == "__main__":
    main()