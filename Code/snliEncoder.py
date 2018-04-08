import time
import os
import torch
from torch import np
import torchvision
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from Models.blocks import *

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

# def getModel(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout):
#     lstm = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout)
#     return lstm

class nliNet(object):
    """docstring for nliNet"""
    def __init__(self, arg):
        super(nliNet, self).__init__()
        self.encoder = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout)

        self.classifier = MLP(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_ln, classifier_dropout_rate)



    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)
        v = self.encoder(s2)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output

    def encode(self, s1):
        emb = self.encoder(s1)
        return emb
        

# class kaggleDataset(Dataset):
#     def __init__(self, csvPath, imagesPath, transform = None):
    
#         self.data = pd.read_csv(csvPath)
#         self.imagesPath = imagesPath
#         self.transform = transform

#         self.imagesData = self.data['image_name']
#         self.labelsData = self.data['tag'].astype('int')

#     def __getitem__(self, index):
#         imageName = os.path.join(self.imagesPath,self.data.iloc[index, 0])
#         tIO1 = time.monotonic()
#         image = Image.open(imageName + '.jpg')
#         tIO2 = time.monotonic()
#         tPre1 = time.monotonic()
#         image = image.convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         tPre2 = time.monotonic()
#         label = self.labelsData[index]
#         return image, label, tIO2-tIO1, tPre2-tPre1 

#     def __len__(self):
#         return len(self.data)


def trainEpoch(epoch, break_val, trainLoader, model, optimizer, criterion):

    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == break_val:
            return
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.data[0]))
            save(model, optimizer, loss, 'snliTrained')


def train(numEpochs, trainLoader, model, optimizer, criterion):
    for epoch in range(numEpochs):
        trainEpoch(epoch,2000,trainLoader,model,optimizer,criterion)


def main():

    batchSize = 100
    epochs = 5
    learningRate = 0.01
    momentum = 0.9
    numWorkers = 1
    
    numEpochs = 5


    inp_dim = 300
    model_dim = 10
    num_layers = 10
    reverse = False
    bidirectional = True
    dropout = 0.1

    model = nliNet()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr = learningRate)
    # optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum)
    optimizer = optim.Adam(model.parameters(), lr = learningRate)

    # imagesPath = 'kaggleamazon/train-jpg/'
    # trainData = 'kaggleamazon/train.csv'
    # testData = 'kaggleamazon/test.csv'

    # transformations = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

    # trainingDataset = kaggleDataset(trainData,imagesPath,transformations)
    # testingDataset = kaggleDataset(testData,imagesPath,transformations)


    # trainLoader = DataLoader(trainingDataset,batchSize,num_workers=numWorkers)
    # testLoader = DataLoader(testingDataset,batchSize,num_workers=numWorkers)


    train(numEpochs, trainLoader, model, optimizer, criterion)



if __name__ == "__main__":
    main()

