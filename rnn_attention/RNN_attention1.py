from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import time
import random
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


# word2index 词到index
# index2word index到词
# word2count 记录word的次数
class Lang:
    def __init__(self, name):
        self.name = name 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'SOS', 1:'EOS'}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False):
    print('Reading lines...')

    f_cn = open('data/%s.txt' % lang1, encoding='utf-8').read().strip().split('\n')
    f_en = open('data/%s.txt' % lang2, encoding='utf-8').read().strip().split('\n')

    pairs = [[s for s in l.split('\t')] for l in f_cn]
    for index, l in enumerate(f_en):
        pairs[index].append(l)

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print('Read %s sentence pairs' % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('cn', 'en')
print(random.choice(pairs))

# Encoder Bidirectional GRU
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, 
        # norm_type=2, scale_grad_by_freq=False, sparse)
        # num_beddings: embeddings字典的大小
        # embedding_dim：embedding vector的大小
        self.embedding = nn.Embedding(input_size, hidden_size)
        # nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, 
        # dropout=0, bidirectional=False)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        # output:[1*1*512], hidden:[2*1*256]
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine =nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # torch.cat(tensors, dim=0, out=None)
        # tensors: 要连接的张量
        # dim：要连接的维度，且该维度必须相同.
        # [1*50]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # torch.bmm(batch1, batch2, out=None)
        # batch1和batch2中的矩阵相乘
        # batch1和bathc2必须是三维
        # encoder_outputs.unsqueeze(0)-- [1*50*512]
        # attn_weights.unsqueeze(0) -- [1*1*50] 
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # attn_applied -- [1*1*512]
        # output -- [1*768]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
    decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.initHidden()
    decoder_hidden = decoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden
        )
        if ei == 1:
            encoder_hidden1 = encoder_hidden
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden1[1:, :, :]


    for di in range(target_length):
        decoder_output , decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # if use_teacher_forcing:

    #     for di in range(target_length):
    #         decoder_output , decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs
    #         )
    #         loss += criterion(decoder_output, target_tensor[di])
    #         decoder_input = target_tensor[di]
    # else:
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_outputs
    #         )
    #         topv, topi = decoder_output.topk(1)
    #         decoder_input = topi.squeeze().detach()

    #         loss += criterion(decoder_output, target_tensor[di])
    #         if decoder_input.item() == EOS_token:
    #             break
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, epoches=10, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # training_pairs = [tensorsFromPair(random.choice(pairs))
    #                   for i in range(n_iters)]
    training_pairs = [tensorsFromPair(pair) for pair in pairs]
    training_pairs_len = len(training_pairs)
    criterion = nn.NLLLoss()
    for epoch in range(epoches):
        for iter in range(1, training_pairs_len + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            # if iter % 100 == 0:
            #     print(input_tensor)
            #     print(target_tensor)
            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%dth epoch: %s (%d %d%%) %.4f' % (epoch+1, timeSince(start, iter / len(training_pairs)),
                                 iter, iter / len(training_pairs) * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        decoder_hidden = decoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size*2, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            if ei == 1:
                encoder_hidden1 = encoder_hidden
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        
        decoder_hidden = encoder_hidden1[1:, :, :]

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
# encoder1 = TheMoudleClass()

encoder1.load_state_dict(torch.load("cn2en_encoder.pkl"))
attn_decoder1.load_state_dict(torch.load("cn2en_attn.pkl"))
# print('training....')
print('Nexttraining....')
trainIters(encoder1, attn_decoder1, epoches=1, print_every=200)
torch.save(encoder1.state_dict(), 'cn2en_encoder.pkl')
torch.save(attn_decoder1.state_dict(), 'cn2en_attn.pkl')
evaluateRandomly(encoder1, attn_decoder1)