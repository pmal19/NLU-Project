import sys
#sys.path.insert(0,"/home/pm2758/etc/text")
import torch
import torch.nn as nn
from torch import optim
import time, random
import os
from tqdm import tqdm
from lstm import LSTMSentiment
from bilstm import BiLSTMSentiment
from torchtext import data
import numpy as np
import argparse
from biLSTMs import *
import pdb

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

def get_accuracy2(tot_correct, tot_samples, label, pred):
    tot_correct += (torch.max(pred, 1)[1].view(label.size()) == label).sum()
    tot_samples += (label.shape[0])
    return tot_correct, tot_samples



def evaluate(model, data, loss_function, name, USE_GPU):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    tot_correct = 0.0
    tot_samples = 0.0
    second=0
    tcorrect=0.0
    for batch in data:
        sent1, sent2, label = batch.text1, batch.text2, batch.label
        maxlen=max(sent1.shape[0], sent2.shape[0])
        if USE_GPU:
            sent1, sent2, label = sent1.cuda(), sent2.cuda(), label.cuda()
        if(sent1.shape[0]==maxlen):
            if USE_GPU:
                sent2=torch.cat([sent2,Variable(torch.ones(maxlen-sent2.shape[0],sent1.shape[1])).long().cuda()])    
            else:
                sent2=torch.cat([sent2,Variable(torch.ones(maxlen-sent2.shape[0],sent1.shape[1])).long()])
        else:
            if USE_GPU:
                sent1=torch.cat([sent1,Variable(torch.ones(maxlen-sent1.shape[0],sent1.shape[1])).long().cuda()])
            else:
                sent1=torch.cat([sent1,Variable(torch.ones(maxlen-sent1.shape[0],sent1.shape[1])).long()])
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        #pdb.set_trace()
        model.hidden = model.init_hidden()
        pred, co = model(sent1, sent2)
        second+=co.data.max(1)[1].sum()
        #pdb.set_trace()
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        tcorrect+=float((pred.max(1)[1]==label).sum())
        avg_loss += loss.data[0]
        #tot_correct, tot_samples = get_accuracy2(tot_correct, tot_samples, label, pred)
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    #tot_samples=
    # acc = tot_correct*100./tot_samples
    # pdb.set_trace()
    tot_samples=len(data)*data.batch_size
    print("SST:", float(second)/tot_samples)
    print(name + ': loss %.2f acc %f' % (avg_loss, tcorrect/tot_samples))
    return acc


def load_quora(text_field, label_field, batch_size):
    train, dev, test = data.TabularDataset.splits(path='./data/Quora/', train='training.full.tsv',
                                                  validation='test.dev.tsv', test='test.dev.tsv', format='tsv',
                                                  fields=[('label', label_field),('text1', text_field), ('text2', text_field)])
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    # train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
    #             batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: data.interleave_keys(len(x.text1),len(x.text2)), repeat=False, device=-1)
    ## for GPU run
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, batch_size, batch_size), sort_key=lambda x: max(len(x.text1),len(x.text2)), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='lstm', help='specify the mode to use (default: lstm)')
args = args.parse_args()

EPOCHS = 20
#USE_GPU = torch.cuda.is_available()
USE_GPU = False
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 32
timestamp = str(int(time.time()))
best_dev_acc = 0.0


text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_quora(text_field, label_field, BATCH_SIZE)
# pdb.set_trace()
model = GumbelQuora(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE, sst_path="", nli_path="", is_direct=True)
if USE_GPU:
    model = model.cuda()
#pdb.set_trace()

print('Load word embeddings...')
# # glove
# text_field.vocab.load_vectors('glove.6B.100d')

# word2vector
word_to_idx = text_field.vocab.stoi
pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
pretrained_embeddings[0] = 0
word2vec = load_bin_vec('/Users/anhadmohananey/Downloads/GoogleNews-vectors-negative300.bin', word_to_idx)
for word, vector in word2vec.items():
    pretrained_embeddings[word_to_idx[word]-1] = vector

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)

model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False
pt=torch.load("/Users/anhadmohananey/Downloads/best_model.pth", map_location="cpu")
#del(pt["embeddings.weight"])
mdict=model.state_dict()
pretrained_dict = {k: v for k, v in pt.items() if k in mdict}
#pdb.set_trace()
mdict.update(pretrained_dict)
#print(pretrained_dict)
model.load_state_dict(mdict)


best_model = model
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda param: param.requires_grad,model.parameters()), lr=1e-3)
loss_function = nn.NLLLoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runsGumbelQuora", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
test_acc = evaluate(best_model, test_iter, loss_function, 'Final Test', USE_GPU)

