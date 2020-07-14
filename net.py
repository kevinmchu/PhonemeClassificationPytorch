# net.py
# Author: Kevin Chu
# Last Modified: 05/20/2020

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, conf_dict):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(conf_dict["num_features"], conf_dict["num_hidden"])
        self.fc2 = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):

    def __init__(self, conf_dict):
        super(CNN, self).__init__()

        self.num_features = conf_dict["num_features"]
        self.window_size = conf_dict["window_size"]
        self.kernel_size = conf_dict["kernel_size"]
        self.max_pool = conf_dict["max_pooling"]
        self.conv1 = nn.Conv2d(2, conf_dict["num_feature_maps"], kernel_size=conf_dict["kernel_size"])

        width = torch.floor(torch.tensor((self.window_size - self.kernel_size[0] + 1)/self.max_pool[0])).to(int)
        height = torch.floor(torch.tensor((self.num_features/2 - self.kernel_size[1] + 1)/self.max_pool[1])).to(int)
        self.fc1 = nn.Linear(width*height*conf_dict["num_feature_maps"], conf_dict["num_hidden"])
        self.fc2 = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])

    def forward(self, x):
        # Add zero padding in time
        x0 = torch.zeros((self.window_size-1, x.size()[1]), dtype=torch.float)
        if torch.cuda.is_available():
            x0 = x0.cuda()
        x = torch.cat((x0, x), dim=0)

        # Separate MFCCs and deltas
        x = torch.transpose(x, 0, 1)
        x = x.view(1, 2, int(self.num_features/2), x.size()[1])

        # Format into feature maps
        batch_sz = x.size()[3] - self.window_size + 1
        idx = torch.linspace(0, self.window_size-1, self.window_size)
        idx = idx.repeat(batch_sz, 1) + torch.linspace(0, batch_sz-1, batch_sz).view(batch_sz, 1)
        idx = idx.to(int)
        x = x[0, :, :, idx]#.view(batch_sz, 2, int(self.num_features/2), self.window_size)
        x = x.permute(2, 0, 1, 3)

        # Pass through network
        x = F.max_pool2d(torch.sigmoid(self.conv1(x)), self.max_pool)
        x = x.view(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class RNNModel(nn.Module):

    def __init__(self, conf_dict):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"], num_layers=1,
                          bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_classes"])

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

    def __init__(self, conf_dict):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"], num_layers=1,
                            bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_classes"])

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


class GRUModel(nn.Module):

    def __init__(self, conf_dict):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"], num_layers=1,
                          bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_classes"])

    def forward(self, x):
        # Reshape features to (num_seq, num_batch, num_feats)
        x = x.view(x.size()[0], 1, x.size()[1])

        # Pass through LSTM layer
        h, _ = self.gru(x)

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


def initialize_network(conf_dict):
    # Instantiate the network
    if conf_dict["model_type"] is "MLP":
        model = MLP(conf_dict)
    elif conf_dict["model_type"] is "CNN":
        model = CNN(conf_dict)
    elif conf_dict["model_type"] is "RNN":
        model = RNNModel(conf_dict)
    elif conf_dict["model_type"] is "LSTM":
        model = LSTMModel(conf_dict)
    elif conf_dict["model_type"] is "GRU":
        model = GRUModel(conf_dict)

    # Initialize weights
    model.apply(initialize_weights)

    return model
