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
        lstm_out1, self.hidden = self.lstmDuplicate(x1, self.hidden)
        lstm_out2, self.hidden = self.lstmDuplicate(x2, self.hidden)
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

class sentiment(nn.Module):
    """docstring for sentiment"""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(sentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.lstmSentiment = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
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
        # x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, _ = self.lstmSentiment(sentence, self.hidden)
        return lstm_out[-1]


class inference(nn.Module):
    """docstring for inference"""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(inference, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.lstmInference = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
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
        # x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, _ = self.lstmInference(sentence, self.hidden)
        return lstm_out[-1]


class duplicate(nn.Module):
    """docstring for duplicate"""
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(duplicate, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.lstmDuplicate = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
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
        # x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, _ = self.lstmDuplicate(sentence, self.hidden)
        return lstm_out[-1]


class GumbelQuora(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, sst_path, nli_path, dropout=0.5):
        super(GumbelQuora, self).__init__()

        loaded = torch.load(sst_path)
        #self.lstmSentiment = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.lstmSentiment = sentiment(embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size)
        newModel = self.lstmSentiment.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        # print(pretrained_dict)
        newModel.update(pretrained_dict)
        self.lstmSentiment.load_state_dict(newModel)
        # print(self.lstmSentiment)
        # print(self.lstmSentiment.lstmSentiment)
        for param in self.lstmSentiment.parameters():
            param.requires_grad = False

        
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


        # self.sst_lstm = load_model("sst", sst_path, embedding_dim, hidden_dim)
        # self.nli_lstm = load_model("nli", nli_path, embedding_dim, hidden_dim)
        #pdb.set_trace()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden2label = nn.Linear(hidden_dim*8, label_size)
        self.g_linear1=nn.Linear(hidden_dim*8, 2)
        # self.hidden = self.init_hidden()

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
        x1 = self.embeddings(sentence1).view(len(sentence1), self.batch_size, -1)
        x2 = self.embeddings(sentence2).view(len(sentence2), self.batch_size, -1)
        # x = torch.cat((x1, x2), 2)
        # sst_out, self.hidden = self.sst_lstm(x, self.hidden)
        # nli_out, self.hidden = self.nli_lstm(x, self.hidden)
        sst_out1 = self.lstmSentiment(x1)
        sst_out2 = self.lstmSentiment(x2)
        sst_out = torch.cat((sst_out1, sst_out2), 1)

        nli_out1 = self.lstmInference(x1)
        nli_out2 = self.lstmInference(x2)
        nli_out = torch.cat((nli_out1, nli_out2), 1)

        g_inp=torch.cat((nli_out, sst_out), 1)
        out_l1=self.g_linear1(g_inp)
        out_l2=F.relu(out_l1)
        out_l3=F.log_softmax(out_l2)
        selector=self.st_gumbel_softmax(out_l3)
        r1 = selector[:,0]
        r2 = selector[:,1]
        r11 = r1.repeat(self.hidden_dim*4,1)
        r22 = r2.repeat(self.hidden_dim*4,1)
        # pdb.set_trace()
        rf = torch.cat((r11,r22),0).transpose(0,1)
        
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
        """
        Return the result of Straight-Through Gumbel-Softmax Estimation.
        It approximates the discrete sampling via Gumbel-Softmax trick
        and applies the biased ST estimator.
        In the forward propagation, it emits the discrete one-hot result,
        and in the backward propagation it approximates the categorical
        distribution via smooth Gumbel-Softmax distribution.

        Args:
            logits (Variable): A un-normalized probability values,
                which has the size (batch_size, num_classes)
            temperature (float): A temperature parameter. The higher
                the value is, the smoother the distribution is.
            mask (Variable, optional): If given, it masks the softmax
                so that indices of '0' mask values are not selected.
                The size is (batch_size, num_classes).

        Returns:
            y: The sampled output, which has the property explained above.
        """
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

# def load_model(task, modelPath, embedding_dim, hidden_dim):
#     loaded = torch.load(modelPath)#'model_state_dict']
#     lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
#     newModel = lstm.state_dict()
#     pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
#     print(pretrained_dict)
#     newModel.update(pretrained_dict)
#     lstm.load_state_dict(newModel)
#     return lstm

def load_model(task, modelPath, embedding_dim, hidden_dim):
    loaded = torch.load(modelPath)#'model_state_dict']
    if task == "sst":
        lstmSentiment = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = lstmSentiment.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        print(pretrained_dict)
        newModel.update(pretrained_dict)
        lstmSentiment.load_state_dict(newModel)
        return lstmSentiment
    if task == "nli":
        lstmInference = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = lstmInference.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        print(pretrained_dict)
        newModel.update(pretrained_dict)
        lstmInference.load_state_dict(newModel)
        return lstmInference
    if task == "quora":
        lstmDuplicate = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        newModel = lstmDuplicate.state_dict()
        pretrained_dict = {k: v for k, v in loaded.items() if k in newModel}
        print(pretrained_dict)
        newModel.update(pretrained_dict)
        lstmDuplicate.load_state_dict(newModel)
        return lstmDuplicate















