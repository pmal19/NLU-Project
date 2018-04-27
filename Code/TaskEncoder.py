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
# import Taskselector as task_selector

import pdb


class TaskEncoder(nn.Module):
    def __init__(self, inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training, sst_path, nli_path, quora_path, which_to_use):
        super(TaskEncoder, self).__init__()
        if which_to_use == "sst":
            loaded = torch.load(sst_path)['model_state_dict']
        elif which_to_use == "quora":
            loaded = torch.load(quora_path)['model_state_dict']
        elif which_to_use == "nli":
        	loaded = torch.load(nli_path)['model_state_dict']
        self.encoderTask = LSTM(inp_dim, model_dim, num_layers, reverse, bidirectional, dropout, training)
        newModel = self.encoderTask.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.encoderTask.load_state_dict(newModel)

    def forward(self, s):
        # print("TaskEncoder forward")
        enc = self.encode(s)
        # pdb.set_trace()
        return enc[-1]

    def encode(self, s1):
        emb = self.encoderTask(s1)
        return emb