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

sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from Models.blocks import *

from data.dataparser import *
from data.batcher import *
from readEmbeddings import *

import pdb

def save(model, optimizer, loss, filename, dev_loss):
    # if the_gpu() >= 0:
    #     recursively_set_device(self.model.state_dict(), gpu=-1)
    #     recursively_set_device(self.optimizer.state_dict(), gpu=-1)

    # Always sends Tensors to CPU.
    save_dict = {
        # 'step': self.step,
        # 'best_dev_error': self.best_dev_error,
        # 'best_dev_step': self.best_dev_step,

        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'vocabulary': self.vocabulary
        'loss': loss.data[0],
        'devloss': dev_loss.data[0]
        }
    # if self.sparse_optimizer is not None:
    #     save_dict['sparse_optimizer_state_dict'] = self.sparse_optimizer.state_dict()
    torch.save(save_dict, filename)

    # if the_gpu() >= 0:
    #     recursively_set_device(self.model.state_dict(), gpu=the_gpu())
    #     recursively_set_device(self.optimizer.state_dict(), gpu=the_gpu())




class sstNet(nn.Module):
    """docstring for sstNet"""
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training):
        super(sstNet, self).__init__()

        self.encoderSst = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training)

        self.classifierSst = MLP(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training)

    def forward(self, s):

        u1 = self.encoderSst(s)
        # pdb.set_trace()
        # features = torch.cat((u1, v1), 2)
        features = u1[-1]
        output = self.classifierSst(features)
        return output

    def encode(self, s):
        emb = self.encoderSst(s)
        return emb
        



class sstDataset(Dataset):
    def __init__(self, sstPath, glovePath, transform = None):
    
        self.data = dataparser.load_sst_data(sstPath)
        self.paddingElement = ['<s>']
        self.maxSentenceLength = self.maxlength(self.data)
        self.vocab = glove2dict(glovePath)

    def __getitem__(self, index):

        s = self.pad(self.data[index]['sentence_1'].split())

        s = self.embed(s)
        
        label = int(self.data[index]['label'])

        return (s), label

    def __len__(self):
        return len(self.data)

    def maxlength(self, data):
        maxSentenceLength = max([len(d['sentence_1'].split()) for d in data])
        return maxSentenceLength

    def pad(self, sentence):
        return sentence + (self.maxSentenceLength-len(sentence))*self.paddingElement

    def embed(self, sentence):
        vector = []
        for word in sentence:
            if str(word) in self.vocab:
                vector = np.concatenate((vector, self.vocab[str(word)]), axis=0)
            else:
                vector = np.concatenate((vector, [0]*len(self.vocab['a'])), axis=0)
        return vector


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
    print("Epoch start - ",epoch)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #pdb.set_trace()
        s = data
        batchSize, _ = s.shape
        s = s.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        if(use_cuda):
            s, target = Variable(s.cuda()), Variable(target.cuda())
        else:
            s, target = Variable(s), Variable(target)
        optimizer.zero_grad()
        output = model(s)
        # pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == break_val:
            return
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0]))
            for (dev_data, dev_target) in enumerate(devLoader):
                sd = dev_data
                devbatchSize, _ = sd.shape
                sd = sd.transpose(0,1).contiguous().view(-1,inp_dim,devbatchSize).transpose(1,2)
                if(use_cuda):
                    sd, dev_target = Variable(sd.cuda()), Variable(dev_target.cuda())
                else:
                    sd, dev_target = Variable(sd), Variable(dev_target)
                    dev_output = model(sd)
                    dev_loss = criterion(dev_output, dev_target)
            save(model, optimizer, loss, 'sstTrained.pth', dev_loss)


def train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize):
    for epoch in range(numEpochs):
        trainEpoch(epoch,20000000,trainLoader,model,optimizer,criterion,inp_dim,batchSize, use_cuda, devLoader, devbatchSize)


def main():

    sstPathTrain = '/scratch/sgm400/NLU_PROJECT/trees/train.txt'
    sstPathDev = '/scratch/sgm400/NLU_PROJECT/trees/dev.txt'
    
    glovePath = '/scratch/sgm400/NLU_PROJECT/glove.840B.300d.txt'

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
    print('Time taken - ',time.time()-t1)
    devDataset = sstDataset(sstPathDev, glovePath)

    devbatchSize = len(devDataset)

    trainLoader = DataLoader(trainingDataset, batchSize, num_workers = numWorkers)
    devLoader = DataLoader(devDataset, devbatchSize, num_workers = numWorkers)

    # for batch_idx, (data, target) in enumerate(trainLoader):
    #     print(batch_idx,' data - ',data,' target - ',target)
    #     print(batch_idx,' data len - ',len(data),' target len - ',len(target))
    #     s1, s2 = data
    #     print(batch_idx,' data len s1 - ',len(s1),' data len s2 - ',len(s2))
    #     print(batch_idx,' data len s1[0] - ',len(s1[0]),' data len s2[0] - ',len(s2[0]))
    #     if batch_idx == 2:
    #         break;
    


    model = sstNet(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training)
    if(use_cuda):
        model.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    # # optimizer = optim.SGD(model.parameters(), lr = learningRate)
    # # optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum)
    # # optimizer = optim.Adam(model.parameters(), lr = learningRate)
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 1e-5)

    train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda, devLoader, devbatchSize)



if __name__ == "__main__":
    main()

