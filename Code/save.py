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

def save(model, optimizer, loss, filename, dev_loss):
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
    torch.save(save_dict, filename)