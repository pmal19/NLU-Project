import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb


class BiLSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstmSentiment = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstmSentiment(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


class BiLSTMInference(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTMInference, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstmInference = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*4, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1, sentence2):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmInference(x1, self.hidden)
        lstm_out2, self.hidden = self.lstmInference(x2, self.hidden)
        # pdb.set_trace()
        lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


class BiLSTMDuplicate(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(BiLSTMDuplicate, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstmDuplicate = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*4, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1, sentence2):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmInference(x1, self.hidden)
        lstm_out2, self.hidden = self.lstmInference(x2, self.hidden)
        # pdb.set_trace()
        lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



class BiLSTMCompSSTonTask(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, sstPath, dropout=0.5):
        super(BiLSTMCompSSTonTask, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        
        loaded = torch.load(sst_path)['model_state_dict']
        self.lstmSentiment = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = self.lstmSentiment.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.lstmSentiment.load_state_dict(newModel)
        # print(pretrained_dict)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2labelMLP = nn.Linear(hidden_dim*4, label_size)
        self.hidden2labelMLP = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1, sentence2):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmSentiment(x1, self.hidden)
        lstm_out2, self.hidden = self.lstmSentiment(x2, self.hidden)
        # pdb.set_trace()
        lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        y = self.hidden2labelMLP(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


class BiLSTMCompNLIonTask1(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, nliPath, dropout=0.5):
        super(BiLSTMCompNLIonTask1, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        
        loaded = torch.load(nliPath)['model_state_dict']
        self.lstmInference = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = self.lstmInference.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.lstmInference.load_state_dict(newModel)
        # print(pretrained_dict)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2labelMLP = nn.Linear(hidden_dim*2, label_size)
        self.hidden2labelMLP = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmInference(x1, self.hidden)
        # pdb.set_trace()
        # lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        lstm_out = lstm_out1
        y = self.hidden2labelMLP(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



class BiLSTMCompNLIonTask2(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, nliPath, dropout=0.5):
        super(BiLSTMCompNLIonTask2, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        
        loaded = torch.load(nliPath)['model_state_dict']
        self.lstmInference = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = self.lstmInference.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.lstmInference.load_state_dict(newModel)
        # print(pretrained_dict)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2labelMLP = nn.Linear(hidden_dim*4, label_size)
        self.hidden2labelMLP = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1, sentence2):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmInference(x1, self.hidden)
        lstm_out2, self.hidden = self.lstmInference(x2, self.hidden)
        # pdb.set_trace()
        lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        y = self.hidden2labelMLP(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs




class BiLSTMCompQuoraonTask1(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, quoraPath, dropout=0.5):
        super(BiLSTMCompQuoraonTask1, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        
        loaded = torch.load(quoraPath)['model_state_dict']
        self.lstmDuplicate = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = self.lstmDuplicate.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.lstmDuplicate.load_state_dict(newModel)
        # print(pretrained_dict)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2labelMLP = nn.Linear(hidden_dim*2, label_size)
        self.hidden2labelMLP = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmDuplicate(x1, self.hidden)
        # pdb.set_trace()
        # lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        lstm_out = lstm_out1
        y = self.hidden2labelMLP(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs



class BiLSTMCompQuoraonTask2(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, quoraPath, dropout=0.5):
        super(BiLSTMCompQuoraonTask2, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        
        loaded = torch.load(quoraPath)['model_state_dict']
        self.lstmDuplicate = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = self.lstmDuplicate.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        newModel.update(pretrained_dict)
        self.lstmDuplicate.load_state_dict(newModel)
        # print(pretrained_dict)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2labelMLP = nn.Linear(hidden_dim*4, label_size)
        self.hidden2labelMLP = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence1, sentence2):
        # pdb.set_trace()
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        lstm_out1, self.hidden = self.lstmDuplicate(x1, self.hidden)
        lstm_out2, self.hidden = self.lstmDuplicate(x2, self.hidden)
        # pdb.set_trace()
        lstm_out = torch.cat((lstm_out1, lstm_out2), 2)
        y = self.hidden2labelMLP(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs