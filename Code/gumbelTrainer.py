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
from sstEncoder import sstNet 
from Models.blocks import *

from data.dataparser import *
from data.batcher import *
from readEmbeddings import *
from Taskselector import *

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
        'dev_loss': dev_loss,
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




class TaskEncoder(nn.Module):
    """docstring for quoraNet"""
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training, sst_path, nli_path, quora_path, which_to_use):
        super(TaskEncoder, self).__init__()
        if which_to_use=="sst":
            loaded = torch.load(sst_path)['model_state_dict']
        elif which_to_use=="quora":
            loaded = torch.load(quora_path)['model_state_dict']
        self.encoderTask = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, False)
        newModel=self.encoderTask.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.encoderTask.load_state_dict(newModel)

    def forward(self, s):
        return self.encode(s)

    def encode(self, s1):
        emb = self.encoderTask(s1)
        return emb

class nliDataset(Dataset):
    def __init__(self, nliPath, glovePath, transform = None):
    
        self.data = dataparser.load_nli_data(nliPath)
        self.paddingElement = ['<s>']
        self.maxSentenceLength = self.maxlength(self.data)
        self.vocab = glove2dict(glovePath)

    def __getitem__(self, index):

        s1 = self.pad(self.data[index]['sentence_1'].split())
        s2 = self.pad(self.data[index]['sentence_2'].split())

        s1 = self.embed(s1)
        s2 = self.embed(s2)
        
        label = LABEL_MAP[self.data[index]['label']]
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

class GumbelNet(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, path1,path2, training=False):
        super(GumbelNet, self).__init__()
        self.encoder1=TaskEncoder(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training, path1, "","", "sst")
        self.encoder2=TaskEncoder(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training, "", "", path2,"quora")
        self.task_selector=Taskselector(model_dim , 2)
        self.classifierNli = MLP(model_dim*4, model_dim, 3, 2, True, dropout, True)
    def forward(self, s1, s2):
        a1=self.encoder1(s1)
        a2=self.encoder1(s1)
        m1=self.task_selector([a1,a2], 2)
        a1=self.encoder1(s2)
        a2=self.encoder1(s2)
        m2=self.task_selector([a1,a2], 2)
        enc_out=torch.cat([m1,m2], dim=2)
        output=self.classifierNli(enc_out)
        return output

def trainEpoch(epoch, break_val, trainLoader, devLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda):
    print("Epoch start - ",epoch)
    for batch_idx, (data, target) in enumerate(trainLoader):
        #pdb.set_trace()
        s1, s2 = data
        s1 = s1.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        s2 = s2.transpose(0,1).contiguous().view(-1,inp_dim,batchSize).transpose(1,2)
        if(use_cuda):
            s1, s2, target = Variable(s1.cuda()), Variable(s2.cuda()), Variable(target.cuda())
        else:
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

            dev_data, dev_target = devLoader
            s1, s2 = dev_data
            s1 = s1.transpose(0,1).contiguous().view(-1,inp_dim,len(devLoader)).transpose(1,2)
            s2 = s2.transpose(0,1).contiguous().view(-1,inp_dim,len(devLoader)).transpose(1,2)

            if(use_cuda):
                s1, s2, dev_target = Variable(s1.cuda()), Variable(s2.cuda()), Variable(dev_target.cuda())
            else:
                s1, s2, dev_target = Variable(s1), Variable(s2), Variable(dev_target)
            
            dev_output = model(s1, s2)
            dev_loss = criterion(dev_output[-1], dev_target)

            save(model, optimizer, loss, 'GumbelsstquoraTry.pth', dev_loss)


def train(numEpochs, trainLoader, devLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda):
    for epoch in range(numEpochs):
        trainEpoch(epoch,20000000,trainLoader,model,optimizer,criterion,inp_dim,batchSize, use_cuda)


def main():

    #quoraPathTrain = '../data/questionsTrain.csv'
    #quoraPathDev = '../data/questionsDev.csv'
    # nliPathTrain="/scratch/am8676/snli_1.0/snli_1.0_train.jsonl"
    # nliPathDev="/scratch/am8676/snli_1.0/snli_1.0_dev.jsonl"
    # glovePath = '/scratch/am8676/glove.840B.300d.txt'

    nliPathTrain="../../Data/snli_1.0/snli_1.0_train.jsonl"
    nliPathDev="../../Data/snli_1.0/snli_1.0_dev.jsonl"
    glovePath = '../../glove.6B/glove.6B.300d.txt'

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
    training = False


    use_cuda = torch.cuda.is_available()


    t1 = time.time()
    devDataset = nliDataset(nliPathDev, glovePath)
    trainingDataset = nliDataset(nliPathTrain, glovePath)
    trainLoader = DataLoader(trainingDataset, batchSize, num_workers = numWorkers)
    devLoader = DataLoader(devDataset, len(devDataset), num_workers = numWorkers)
    print('DataLoading time taken - ',time.time()-t1)

    sst_path="sstTrained"
    nli_path="quoraTrained"
    quora_path="quoraTrained"
    #self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training=True
    model = GumbelNet(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, sst_path, quora_path, training)
    if(use_cuda):
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    # # optimizer = optim.SGD(model.parameters(), lr = learningRate)
    # # optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum)
    # # optimizer = optim.Adam(model.parameters(), lr = learningRate)
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = 1e-5)

    train(numEpochs, trainLoader, devLoader, model, optimizer, criterion, inp_dim, batchSize, use_cuda)



if __name__ == "__main__":
    main()

