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

sys.path.insert(0, '../../')
sys.path.insert(0, '../')

from Models.blocks import *

from data.dataparser import *
from data.batcher import *
from readEmbeddings import *

import pdb

def save(model, optimizer, loss, filename):
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
        'loss': loss.data[0]
        }
    # if self.sparse_optimizer is not None:
    #     save_dict['sparse_optimizer_state_dict'] = self.sparse_optimizer.state_dict()
    torch.save(save_dict, filename)

    # if the_gpu() >= 0:
    #     recursively_set_device(self.model.state_dict(), gpu=the_gpu())
    #     recursively_set_device(self.optimizer.state_dict(), gpu=the_gpu())




class quoraNet(nn.Module):
    """docstring for quoraNet"""
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training):
        super(quoraNet, self).__init__()

        self.encoderQuora = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training)

        self.classifierQuora = MLP(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training)

    def forward(self, s1, s2):

        u1 = self.encoderQuora(s1)
        v1 = self.encoderQuora(s2)
        # pdb.set_trace()
        features = torch.cat((u1, v1), 2)
        output = self.classifierQuora(features)
        return output

    def encode(self, s1):
        emb = self.encoderQuora(s1)
        return emb
        



class qoraDataset(Dataset):
    def __init__(self, nliPath, glovePath, transform = None):
    
        self.data = dataparser.load_quora_data(nliPath)
        self.paddingElement = ['<s>']
        self.maxSentenceLength = self.maxlength(self.data)
        self.vocab = glove2dict(glovePath)

    def __getitem__(self, index):

        s1 = self.pad(self.data[index]['sentence_1'].split())
        s2 = self.pad(self.data[index]['sentence_2'].split())

        s1 = self.embed(s1)
        s2 = self.embed(s2)
        
        label = self.data[index]['label']
        return (s1, s2), label

    def __len__(self):
        return len(self.data)

    def maxlength(self, data):
        maxSentenceLength = max([max(len(d['sentence_1'].split()),len(d['sentence_2'].split())) for d in data])
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


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion, inp_dim, batchSize):
    print("Epoch start - ",epoch)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #pdb.set_trace()
        s1, s2 = data
        s1 = s1.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        s2 = s2.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        s1, s2, target = Variable(s1), Variable(s2), Variable(target)
        optimizer.zero_grad()
        output = model(s1, s2)
        # pdb.set_trace()
        loss = criterion(output[-1], target)
	print(batch_idx,loss.data[0])
        loss.backward()
        optimizer.step()
        if batch_idx == break_val:
            return
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0]))
            save(model, optimizer, loss, 'quoraTrained2')


def train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize):
    for epoch in range(numEpochs):
        trainEpoch(epoch,20000000,trainLoader,model,optimizer,criterion,inp_dim,batchSize)


def main():

    quoraPathTrain = '../data/questionsTrain.csv'
    quoraPathDev = '../data/questionsDev.csv'
    
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
    trainingDataset = qoraDataset(quoraPathTrain, glovePath)
    print('Time taken - ',time.time()-t1)
    # devDataset = qoraDataset(nliPathDev, glovePath)

    trainLoader = DataLoader(trainingDataset, batchSize, num_workers = numWorkers)
    # devLoader = DataLoader(testingDataset, battrainLoader = DataLoader(trainingDataset, batchSize, num_workers = numWorkers)chSize, num_workers = numWorkers)

    # for batch_idx, (data, target) in enumerate(trainLoader):
    #     print(batch_idx,' data - ',data,' target - ',target)
    #     print(batch_idx,' data len - ',len(data),' target len - ',len(target))
    #     s1, s2 = data
    #     print(batch_idx,' data len s1 - ',len(s1),' data len s2 - ',len(s2))
    #     print(batch_idx,' data len s1[0] - ',len(s1[0]),' data len s2[0] - ',len(s2[0]))
    #     if batch_idx == 2:
    #         break;
    


    model = quoraNet(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate, training)

    criterion = nn.CrossEntropyLoss()
    # # optimizer = optim.SGD(model.parameters(), lr = learningRate)
    # # optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum)
    # # optimizer = optim.Adam(model.parameters(), lr = learningRate)
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 1e-5)

    train(numEpochs, trainLoader, model, optimizer, criterion, inp_dim, batchSize)



if __name__ == "__main__":
    main()

