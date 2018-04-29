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
import torch.backends.cudnn as cudnn
#from torchvision import transforms, utils
# from itertools import accumulate
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from Models.blocks import *

from data.dataparser import *
from data.batcher import *
from readEmbeddings import *
from datasets import *
from save import *
from TaskEncoder import *

import pdb

class ClassLSTM(nn.Module):
    """docstring for ClassLSTM"""
    def __init__(self, input_size, hidden_size, num_layers, batch, bias = True, batch_first = False, dropout = 0, bidirectional = False):
        super(ClassLSTM, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias = True, batch_first = False, dropout = dropout, bidirectional = False)
        self.h0 = Variable(torch.randn(num_layers * self.num_directions, batch, hidden_size))
        self.c0 = Variable(torch.randn(num_layers * self.num_directions, batch, hidden_size))
    def forward(self, s1):
        output, hn = self.lstm(s1, (self.h0, self.c0))
        return output
        


class sstNet(nn.Module):
    """docstring for sstNet"""
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training, batchSize):
        super(sstNet, self).__init__()
        # self.num_layers=num_layers
        # self.model_dim=model_dim
        # self.training=training
        # self.v_rnn=nn.LSTM(inp_dim, model_dim, num_layers=num_layers, batch_first=True)
        # self.encoderSst = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training)
        self.encoderSst = ClassLSTM(inp_dim, model_dim, num_layers, batchSize, bidirectional = bidirectional, dropout = dropout)
        self.classifierSst = MLP(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training)

    def forward(self, s1):
        # batch_size, seq_len = s.size()[:2]
        # h0=Variable(
        #     torch.zeros(self.num_layers,
        #          batch_size, self.model_dim), requires_grad=not self.training
        #     )
        # c0=Variable(
        #     torch.zeros(self.num_layers,
        #          batch_size, self.model_dim), requires_grad=not self.training
        #     )
        # o, (hn, _) = self.v_rnn(s.float(), (h0, c0))

        # hn, output = self.encoderSst(s1)
        oE = self.encoderSst(s1)
        #features = hn.squeeze()
        features = oE[-1]
        output = F.log_softmax(self.classifierSst(features))
        # pdb.set_trace()
        return output

    def encode(self, s):
        emb = self.encoderSst(s)
        return emb


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
    print("Epoch start - ",epoch)
    for batch_idx, (data, target) in enumerate(trainLoader):
        s1 = data.float()
        batch, _ = s1.shape
        if batchSize != batch:
            break
        s1 = s1.transpose(0,1).contiguous().view(-1,inp_dim,batch).transpose(1,2)
        if(use_cuda):
            s1, target = Variable(s1.cuda()), Variable(target.cuda())
        else:
            s1, target = Variable(s1), Variable(target)
        # pdb.set_trace()
        output = model(s1)
        model.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == break_val:
            return
        if batch_idx % 100 == 0:
            dev_loss = 0
            n_correct = 0
            n_total = 0
            for idx, (dev_data, dev_target) in enumerate(devLoader):
                sd = dev_data.float()
                # pdb.set_trace()
                devbatchSize, _ = sd.shape
                if batchSize != devbatchSize:
                    break
                sd = sd.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
                if(use_cuda):
                    sd, dev_target = Variable(sd.cuda()), Variable(dev_target.cuda())
                else:
                    sd, dev_target = Variable(sd), Variable(dev_target)
                dev_output = model(sd)
                dev_loss += criterion(dev_output, dev_target)
		# if idx == 0:
		#	print(dev_output)
                n_correct += (torch.max(dev_output, 1)[1].view(dev_target.size()) == dev_target).sum()
                n_total += devbatchSize
                # break
            dev_acc = (100. * n_correct.data[0])/n_total

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDev Loss: {:.6f}\tDev Acc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0], dev_loss.data[0], dev_acc))
            save(model, optimizer, loss, 'sstTrained.pth', dev_loss, dev_acc)
    return loss


def train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
    for epoch in range(numEpochs):
        loss = trainEpoch(epoch,20000000,trainLoader,model,optimizer,criterion,inp_dim,batchSize, use_cuda, devLoader, devbatchSize)
        dev_loss = 0
        n_correct = 0
        n_total = 0
        for idx, (dev_data, dev_target) in enumerate(devLoader):
            sd = dev_data
            # pdb.set_trace()
            devbatchSize, _ = sd.shape
            sd = sd.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
            if(use_cuda):
                sd, dev_target = Variable(sd.cuda()), Variable(dev_target.cuda())
            else:
                sd, dev_target = Variable(sd), Variable(dev_target)
            dev_output = model(sd)
            dev_loss += criterion(dev_output, dev_target)
            n_correct += (torch.max(dev_output, 1)[1].view(dev_target.size()) == dev_target).sum()
            n_total += devbatchSize
        dev_acc = (100. * n_correct.data[0])/n_total
        print('Epoch: {} - Dev Accuracy: {}'.format(epoch, dev_acc))
        save(model, optimizer, loss, 'sstTrainedEpoch.pth', dev_loss, dev_acc)


def main():

    local=False
    if(local):
        quoraPathTrain = '../../data/questionsTrain.csv'
        quoraPathDev = '../../data/questionsDev.csv'
        nliPathTrain = "../../Data/snli_1.0/snliSmallaa"
        nliPathDev = "../../Data/snli_1.0/snliSmallDevaa"
        sstPathTrain = "../../../trees/train.txt"
        sstPathDev = "../../../trees/dev.txt"
        glovePath = '../../../glove.6B/glove.6B.300d.txt'
    else:
        quoraPathTrain = '../../data/questionsTrain.csv'
        quoraPathDev = '../../data/questionsDev.csv'
        nliPathTrain = "/scratch/pm2758/nlu/snli_1.0/snli_1.0_train.jsonl"
        nliPathDev = "/scratch/pm2758/nlu/snli_1.0/snli_1.0_dev.jsonl"
        sstPathTrain = "/scratch/pm2758/nlu/trees/train.txt"
        sstPathDev = "/scratch/pm2758/nlu/trees/dev.txt"
        glovePath = '/scratch/pm2758/nlu/glove.840B.300d.txt'

    batchSize = 64
    learningRate = 0.001
    momentum = 0.9
    numWorkers = 5
    
    numEpochs = 10

    inp_dim = 300
    model_dim = 300
    num_layers = 1
    reverse = False
    bidirectional = False
    dropout = 0.1

    mlp_input_dim = 300
    mlp_dim = 300
    num_classes = 5
    num_mlp_layers = 2
    mlp_ln = True
    classifier_dropout_rate = 0.1

    training = True

    use_cuda = torch.cuda.is_available()
    if(use_cuda):
        the_gpu.gpu = 0

    t1 = time.time()
    trainingDataset = sstDataset(sstPathTrain, glovePath)
    devDataset = sstDataset(sstPathDev, glovePath, training = False)
    print('Time taken - ',time.time()-t1)
    devbatchSize = batchSize

    trainLoader = DataLoader(trainingDataset, batchSize, shuffle=False, num_workers = numWorkers)
    devLoader = DataLoader(devDataset, devbatchSize, shuffle=False, num_workers = numWorkers)

    model = sstNet(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training, batchSize)
    if(use_cuda):
        model.cuda()
    if(use_cuda):
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion=nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 1e-5)

    train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize)



if __name__ == "__main__":
    main()
