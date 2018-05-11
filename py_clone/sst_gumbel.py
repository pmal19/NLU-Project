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





class GumbelSSTwithNews(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, nli_path, quora_path, news_path, dropout=0.5):
        super(GumbelSSTwithNews, self).__init__()

        loaded = torch.load(nli_path)
        # self.lstmInference = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.lstmInference = inference(embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size)
        newModel = self.lstmInference.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        # print(pretrained_dict)
        newModel.update(pretrained_dict)
        self.lstmInference.load_state_dict(newModel)
        # print(self.lstmInference)
        # print(self.lstmInference.lstmInference)
        for param in self.lstmInference.parameters():
            param.requires_grad = False

        
        loaded = torch.load(quora_path)
        # self.lstmDuplicate = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.lstmDuplicate = duplicate(embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size)
        newModel = self.lstmDuplicate.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        # print(pretrained_dict)
        newModel.update(pretrained_dict)
        self.lstmDuplicate.load_state_dict(newModel)
        # print(self.lstmDuplicate)
        # print(self.lstmDuplicate.lstmDuplicate)
        for param in self.lstmDuplicate.parameters():
            param.requires_grad = False


        loaded = torch.load(news_path)
        # self.lstmNews = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.lstmNews = duplicate(embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size)
        newModel = self.lstmNews.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        # print(pretrained_dict)
        newModel.update(pretrained_dict)
        self.lstmNews.load_state_dict(newModel)
        # print(self.lstmNews)
        # print(self.lstmNews.lstmNews)
        for param in self.lstmNews.parameters():
            param.requires_grad = False


        # self.sst_lstm = load_model("sst", sst_path, embedding_dim, hidden_dim)
        # self.nli_lstm = load_model("nli", nli_path, embedding_dim, hidden_dim)
        #pdb.set_trace()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2label = nn.Linear(hidden_dim*6, label_size)
        self.g_linear1=nn.Linear(hidden_dim*6, 2)

    def forward(self, sentence1):
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        # x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        # sst_out, self.hidden = self.sst_lstm(x, self.hidden)
        # nli_out, self.hidden = self.nli_lstm(x, self.hidden)
        sst_out = self.lstmInference(x1)
        # sst_out2 = self.lstmInference(x2)
        # sst_out = torch.cat((sst_out1, sst_out2), 1)

        nli_out = self.lstmDuplicate(x1)
        # nli_out2 = self.lstmDuplicate(x2)
        # nli_out = torch.cat((nli_out1, nli_out2), 1)

        news_out = self.lstmNews(x1)
        # nli_out2 = self.lstmDuplicate(x2)
        # nli_out = torch.cat((nli_out1, nli_out2), 1)

        g_inp=torch.cat((nli_out, sst_out, news_out), 1)
        out_l1=self.g_linear1(g_inp)
        out_l2=F.relu(out_l1)
        out_l3=F.log_softmax(out_l2)
        selector=self.st_gumbel_softmax(out_l3)
        r1 = selector[:,0]
        r2 = selector[:,1]
        r3 = selector[:,2]
        r11 = r1.repeat(self.hidden_dim*2,1)
        r22 = r2.repeat(self.hidden_dim*2,1)
        r33 = r2.repeat(self.hidden_dim*2,1)
        # pdb.set_trace()
        rf = torch.cat((r11,r22,r33),0).transpose(0,1)
        
        ret = g_inp*rf
        y = self.hidden2label(ret)
        log_probs = F.log_softmax(y)
        return log_probs, selector


    def masked_softmax(self,logits, mask=None):
        eps = 1e-20
        probs = F.softmax(logits)
        if mask is not None:
            mask = mask.float()
            probs = probs * mask + eps
            probs = probs / probs.sum(1, keepdim=True)
        return probs

    def st_gumbel_softmax(self,logits, temperature=1.0, mask=None):
        def convert_to_one_hot(indices, num_classes):
            batch_size = indices.size(0)
            indices = indices.unsqueeze(1)
            one_hot = Variable(indices.data.new(batch_size, num_classes).zero_().scatter_(1, indices.data, 1))
            return one_hot

        eps = 1e-20
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = Variable(-torch.log(-torch.log(u + eps) + eps))
        y = logits + gumbel_noise
        y = self.masked_softmax(logits=y / temperature, mask=mask)
        y_argmax = y.max(1)[1]
        # pdb.set_trace()
        y_hard = convert_to_one_hot(
            indices=y_argmax,
            num_classes=y.size(1)).float()
        y = (y_hard - y).detach() + y
        return y






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
    tot_correct += long((torch.max(pred, 1)[1].view(label.size()) == label).sum())
    tot_samples += long((label.shape[0]))
    return tot_correct, tot_samples


def train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch, USE_GPU):
    model.train()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    tot_correct = 0.0
    tot_samples = 0.0
    count = 0
    for batch in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        sent, label = batch.text, batch.label
        if USE_GPU:
            sent, label = sent.cuda(), label.cuda()
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        # model.hidden = model.init_hidden()
        pred = model(sent)
        # pred_label = pred.data.max(1)[1].numpy()
        # pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        loss.backward()
        optimizer.step()
        tot_correct, tot_samples = get_accuracy2(tot_correct, tot_samples, label, pred)
    avg_loss /= len(train_iter)
    # acc = get_accuracy(truth_res, pred_res)
    acc = tot_correct/tot_samples
    return avg_loss, acc


def evaluate(model, data, loss_function, name, USE_GPU):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    tot_correct = 0.0
    tot_samples = 0.0
    for batch in data:
        sent, label = batch.text, batch.label
        if USE_GPU:
            sent, label = sent.cuda(), label.cuda()
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        # model.hidden = model.init_hidden()
        pred = model(sent)
        # pred_label = pred.data.max(1)[1].numpy()
        # pred_res += [x for x in pred_label]
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        tot_correct, tot_samples = get_accuracy2(tot_correct, tot_samples, label, pred)
    avg_loss /= len(data)
    # acc = get_accuracy(truth_res, pred_res)
    acc = tot_correct*100./tot_samples
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc


def load_sst(text_field, label_field, batch_size):
    train, dev, test = data.TabularDataset.splits(path='./data/SST2/', train='train.tsv',
                                                  validation='dev.tsv', test='test.tsv', format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, dev, test)
    label_field.build_vocab(train, dev, test)
    # train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
    #             batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=-1)
    ## for GPU run
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                batch_sizes=(batch_size, len(dev), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=None)
    return train_iter, dev_iter, test_iter


# def adjust_learning_rate(learning_rate, optimizer, epoch):
#     lr = learning_rate * (0.1 ** (epoch // 10))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return optimizer


args = argparse.ArgumentParser()
args.add_argument('--m', dest='model', default='lstm', help='specify the mode to use (default: lstm)')
args = args.parse_args()

EPOCHS = 20
USE_GPU = torch.cuda.is_available()
EMBEDDING_DIM = 300
HIDDEN_DIM = 150

BATCH_SIZE = 32
timestamp = str(int(time.time()))
best_dev_acc = 0.0


text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter, test_iter = load_sst(text_field, label_field, BATCH_SIZE)

quora_path = "best_model_quora/best_model.pth"
sst_path = "best_model_sst/best_model.pth"
nli_path="best_model_nli/best_model.pth"
news_path="best_model_news/best_model.pth"

# pdb.set_trace()
model = GumbelSSTwithNews(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, vocab_size=len(text_field.vocab), label_size=len(label_field.vocab)-1,\
                          use_gpu=USE_GPU, batch_size=BATCH_SIZE, nli_path=nli_path, quora_path=quora_path, news_path=news_path)

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
word2vec = load_bin_vec('./data/GoogleNews-vectors-negative300.bin', word_to_idx)
for word, vector in word2vec.items():
    pretrained_embeddings[word_to_idx[word]-1] = vector

# text_field.vocab.load_vectors(wv_type='', wv_dim=300)

model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
# model.embeddings.weight.data = text_field.vocab.vectors
# model.embeddings.embed.weight.requires_grad = False


best_model = model
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda param: param.requires_grad,model.parameters()), lr=1e-3)
loss_function = nn.NLLLoss()

print('Training...')
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runsGumbelSST", timestamp))
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
for epoch in range(EPOCHS):
    avg_loss, acc = train_epoch_progress(model, train_iter, loss_function, optimizer, text_field, label_field, epoch, USE_GPU)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
    torch.save(model.state_dict(), out_dir + '/best_model' + '.pth')
    # dev_acc = evaluate(model, dev_iter, loss_function, 'Dev', USE_GPU)
    # if dev_acc > best_dev_acc:
    #     if best_dev_acc > 0:
    #         os.system('rm '+ out_dir + '/best_model' + '.pth')
    #     best_dev_acc = dev_acc
    #     best_model = model
    #     torch.save(best_model.state_dict(), out_dir + '/best_model' + '.pth')
    #     # evaluate on test with the best dev performance model
    #     test_acc = evaluate(best_model, test_iter, loss_function, 'Test', USE_GPU)
test_acc = evaluate(best_model, test_iter, loss_function, 'Final Test', USE_GPU)

