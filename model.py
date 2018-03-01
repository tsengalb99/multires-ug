from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Parameter
from torch.autograd import Variable
from torch.sparse import FloatTensor as STensor
from torch.cuda.sparse import FloatTensor as CudaSTensor
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from varGen import * #generate variance to replace the old variance fed in by data

import util

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, args=None):
        print("INPUT SIZE" + str(input_size))
        super(EncoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size) #input_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, args=None):
        super(DecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        print(hidden_size)
        self.embedding = nn.Linear(output_size, hidden_size) #output_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)#output_size
        # self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, var):
        embedded = self.embedding(input)
        embedded = F.relu(embedded)
        embedded = torch.unsqueeze(embedded, 0)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output.squeeze(0))

        #corrects for old variance (effective removes it)
        #decoder outputs predicted based on original variance, we will change this
        #        print(output)
        #        for i in range(len(output)):
        #            vx, vy, vz = varGen(output[i][0], output[i][1], output[i][2], 1)
        #            vA = [vx, vy, vz]
        #            vAL = [vx.data, vy.data, vz.data]
        #            print(vAL, var[i].data)
        #            for j in range(3):
        #                if(float(var[i][j]) != 0):
        #                    output[i, j] = output[i,j] * var[i][j]
        #                    output[i, j] = output[i,j] / vA[j]                 
        #        print(output)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, args=None):
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(output_size, hidden_size)#output_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        self.attn = nn.Linear(self.hidden_size * 2, self.args.output_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size * 2, output_size)#output_size

        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, encoder_outputs):
        # input  B x H
        # hidden 1 x B x H
        # output B x H

        embedded = F.relu(self.embedding(input))

        # Calculate attention weights and apply to encoder outputs

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden.squeeze(0)), 1))) # B x T

        print(attn_weights.size(), "attn_weights", encoder_outputs.size(), "encoder_outputs")

        # B x 1 x T * B x T x H = B x 1 x H = B x H
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs.transpose(0, 1)).squeeze(1)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = self.attn_combine(torch.cat((embedded, context), 1))
        rnn_input = rnn_input.unsqueeze(0)

        output, hidden = self.gru(rnn_input, hidden)

        output = self.softmax(self.out( torch.cat((output.squeeze(0), context), 1) ))

        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, self.args.batch_size, self.hidden_size))
        if self.args.cuda:
            return result.cuda()
        else:
            return result


class Seq2seq(nn.Module):
    def __init__(self, args, dims, test=False):
        super(Seq2seq, self).__init__()
        self.args = args

        T = torch.cuda if self.args.cuda else torch
        print("args state dim", self.args.state_dim)
        self.enc = EncoderRNN(self.args.state_dim, self.args.hidden_size, args=args)

        self.use_attn = False

        if self.use_attn:
            self.dec = AttnDecoderRNN(self.args.hidden_size, self.args.state_dim, args=args)
        else:
            self.dec = DecoderRNN(self.args.hidden_size, self.args.state_dim, args=args)

    def parameters(self):
        return list(self.enc.parameters()) + list(self.dec.parameters())

    def forward(self, x, ytarget):

 #       print("recieved data")
        encoder_hidden = self.enc.initHidden()

        hs = []
        for t in range(self.args.input_len):
            encoder_output, encoder_hidden = self.enc(x[t], encoder_hidden)
            hs += [encoder_output]
#        print(hs)
        decoder_hidden = hs[-1]

        hs = torch.cat(hs, 0)

        inp = Variable(torch.zeros(self.args.batch_size, self.args.state_dim))
        if self.args.cuda: inp = inp.cuda()
        ys = []

        if self.use_attn:
            for t in range(self.args.output_len):
                decoder_output, decoder_hidden = self.dec(inp, decoder_hidden, hs)
                inp = decoder_output
                ys += [decoder_output]
        else:
#            print(len(x))
            for t in range(self.args.output_len):
                decoder_output, decoder_hidden = self.dec(inp, decoder_hidden, ytarget[t,:,3:])
                inp = decoder_output
                ys += [decoder_output]
        out = torch.cat([torch.unsqueeze(y, dim=0) for y in ys])
        return out