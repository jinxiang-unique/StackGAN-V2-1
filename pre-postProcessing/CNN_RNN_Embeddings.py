import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pickle
from nltk.tokenize import RegexpTokenizer

WORDS_NUM =  18

class Embeddings():

    def __init__(self):

        # load word dictary
        with open(r"data/captions.pickle", 'rb') as f:
            x = pickle.load(f)
            self.wordtoix = x[3]
            del x

        # get the gpu
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.device("cuda:0"))
        else:
            raise Exception("Get cuda you loser")

        # load text encoder
        self.textEncoder = RNN_ENCODER(len(self.wordtoix), nhidden=256).cuda()
        self.textEncoder.load_state_dict(torch.load(r"models/text_encoder200.pth"))
        self.textEncoder.eval()

    def tokenize(self, cap):

        cap = cap.replace("\ufffd\ufffd", " ")
        # picks out sequences of alphanumeric characters as tokens
        # and drops everything else
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cap.lower())

        assert len(tokens) != 0, "should not be zero"

        tokens_new = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0:
                tokens_new.append(t)

        return tokens_new

    def embedSentenceBatch(self, sentences):

        # split the sentences into seprate words e.g. ["hello world"] to ["hello", "world"]
        for i in range(len(sentences)):
            #sentences = sentences.lower().split()
            sentences[i] = self.tokenize(sentences[i])

        # turns the words into numbers e.g. ["hello", "world"] to ["1", "2"]
        tokenLens = []
        for i in range(len(sentences)):

            tokens = []
            for word in sentences[i]:
                if word in self.wordtoix:
                    tokens.append(self.wordtoix[word])

                    if len(tokens) == WORDS_NUM:
                        break
                else:
                    print (f"'{word}' not in dict")

            sentences[i] = tokens
            tokenLens.append(len(tokens))

        # pads the tokenized sentences to WORDS_NUM e.g. ["1", "2"] to ["1", "2", "0", "0"]
        paddedTokens = np.zeros((10, WORDS_NUM), dtype="int64")
        for i in range(len(sentences)):

            paddedTokens[i, :tokenLens[i]] = sentences[i]

        # turns the tokenized sentences the sentences embeddings
        with torch.no_grad():

            tokens = torch.LongTensor(paddedTokens).cuda()
            tokenLens = torch.LongTensor(tokenLens).cuda()

            sortedTokenLens, sortedTokenIndices = torch.sort(tokenLens, 0, True)
            tokens = tokens[sortedTokenIndices]

            hidden = self.textEncoder.init_hidden(10)
            words_emb, sent_emb = self.textEncoder(tokens, sortedTokenLens, hidden)

            return sent_emb.cpu().numpy()

    def embedSentence(self, sentence):

        # split the sentence into seprate words e.g. ["hello world"] to ["hello", "world"]
        #sentence = sentence.lower().split()
        sentence = self.tokenize(sentence)

        # turns the words into numbers e.g. ["hello", "world"] to ["1", "2"]
        tokens = []
        for word in sentence:
            if word in self.wordtoix:
                tokens.append(self.wordtoix[word])

                if len(tokens) == WORDS_NUM:
                    break
            else:
                print (f"'{word}' not in dict")

        tokenLen = len(tokens)

        # pads the tokenized sentence to WORDS_NUM e.g. ["1", "2"] to ["1", "2", "0", "0"]
        paddedTokens = np.zeros((1, WORDS_NUM), dtype="int64")
        paddedTokens[0, :tokenLen] = tokens

        # turns the tokenized sentences the sentences embeddings
        with torch.no_grad():

            tokens = torch.LongTensor(paddedTokens).cuda()
            tokenLen = torch.LongTensor([tokenLen]).cuda()

            hidden = self.textEncoder.init_hidden(1)
            words_emb, sent_emb = self.textEncoder(tokens, tokenLen, hidden)

            return sent_emb.cpu().numpy()

# Text2Image Encoder-Decoder
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                    nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                                self.nlayers, batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                                self.nlayers, batch_first=True,
                                dropout=self.drop_prob,
                                bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb
