# net.py
# Author: Kevin Chu
# Last Modified: 05/20/2020

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, num_features, num_hidden, num_classes):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_classes)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


# class RNNModel(nn.Module):
#
#     def __init__(self, num_features, num_hidden, num_classes, bidirectional=False):
#         super(RNNModel, self).__init__()
#
#         # Properties
#         self.num_hidden = num_hidden
#         self.bidirectional = bidirectional
#
#         # Inputs to the hidden layer
#         self.ih_f = nn.Linear(num_features, self.num_hidden)
#         self.hh_f = nn.Linear(self.num_hidden, self.num_hidden)
#
#         # Add backward hidden layer for bidirectional nets
#         if self.bidirectional:
#             self.ih_b = nn.Linear(num_features, self.num_hidden)
#             self.hh_b = nn.Linear(self.num_hidden, self.num_hidden)
#             self.fc = nn.Linear(2*self.num_hidden, num_classes)
#         else:
#             self.fc = nn.Linear(self.num_hidden, num_classes)
#
#     def forward(self, x):
#         if not self.bidirectional:
#             h = self.unidirectional_pass(x)
#         else:
#             h_forward = self.unidirectional_pass(x)
#             h_backward = self.unidirectional_pass(x, "backward")
#             h = torch.cat((h_forward, h_backward), 1)
#
#         # Pass to output layer
#         out = self.fc(h)
#         out = F.log_softmax(out, dim=1)
#
#         return out
#
#     def unidirectional_pass(self, x, direction="forward"):
#         # Hidden unit activations
#         # First row contains initial hidden activations
#         h = torch.zeros(x.size()[0]+1, self.num_hidden)
#         #h = h.to(get_device())
#
#         # Recursively calculate hidden unit activations
#         for i in range(1, x.size()[0]+1):
#             if direction is "forward":
#                 h[i, :] = torch.sigmoid(self.ih_f(x[i-1, :]) + self.hh_f(h[i-1, :].clone()))
#             else:
#                 h[i, :] = torch.sigmoid(self.ih_b(x[-i, :]) + self.hh_b(h[i-1, :].clone()))
#
#         # Ignore h0 activations
#         return h[1:, :]


class RNNModel(nn.Module):

    def __init__(self, num_features, num_hidden, num_classes, bidirectional=False):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size=num_features, hidden_size=num_hidden, num_layers=1, bidirectional=bidirectional)

        # If bidirectional, double number of hidden units
        if not bidirectional:
            self.fc = nn.Linear(num_hidden, num_classes)
        else:
            self.fc = nn.Linear(2*num_hidden, num_classes)

    def forward(self, x):
        # Reshape features to (num_seq, num_batch, num_feats)
        x = x.view(x.size()[0], 1, x.size()[1])

        # Pass through recurrent layer
        h, _ = self.rnn(x)

        # Reshape hidden features to (num_seq, num_feats)
        h = h.view(h.size()[0], h.size()[2])

        # Invert tanh and apply sigmoid
        h = 1/(1 + torch.sqrt((1 - h)/(1 + h)))

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=1)

        return out


class LSTMModel(nn.Module):

    def __init__(self, num_features, num_hidden, num_classes, bidirectional=False):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=num_features, hidden_size=num_hidden, num_layers=1, bidirectional=bidirectional)

        # If bidirectional, double number of hidden units
        if not bidirectional:
            self.fc = nn.Linear(num_hidden, num_classes)
        else:
            self.fc = nn.Linear(2*num_hidden, num_classes)

    def forward(self, x):
        # Reshape features to (num_seq, num_batch, num_feats)
        x = x.view(x.size()[0], 1, x.size()[1])

        # Pass through LSTM layer
        h, (_, _) = self.lstm(x)

        # Reshape hidden features to (num_seq, num_feats)
        h = h.view(h.size()[0], h.size()[2])

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=1)

        return out


def initialize_weights(m):
    """ Initialize weights from Uniform(-0.1,0.1) distribution
    as was done in Graves and Schmidhuber, 2005
    
    Args:
        m
        
    Returns:
        none
    """
    a = -0.1
    b = 0.1

    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data, a=a, b=b)
        nn.init.uniform_(m.bias.data, a=a, b=b)

    if isinstance(m, nn.RNN) or isinstance(m, nn.LSTM):
        nn.init.uniform_(m.weight_ih_l0, a=a, b=b)
        nn.init.uniform_(m.weight_hh_l0, a=a, b=b)
        nn.init.uniform_(m.bias_ih_l0, a=a, b=b)
        nn.init.uniform_(m.bias_hh_l0, a=a, b=b)
