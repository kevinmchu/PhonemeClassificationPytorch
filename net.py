# net.py
# Author: Kevin Chu
# Last Modified: 08/03/2020

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxRegression(nn.Module):

    def __init__(self, conf_dict):
        super(SoftmaxRegression, self).__init__()

        self.num_features = conf_dict["num_features"]
        self.window_size = conf_dict["window_size"]
        self.fc = nn.Linear(self.num_features*self.window_size, conf_dict["num_classes"])

    def forward(self, x):
        x = self.fc(x)

        return F.log_softmax(x, dim=2)


class MLP(nn.Module):

    def __init__(self, conf_dict):
        super(MLP, self).__init__()

        self.num_features = conf_dict["num_features"]
        self.window_size = conf_dict["window_size"]
        self.fc1 = nn.Linear(conf_dict["window_size"]*conf_dict["num_features"], conf_dict["num_hidden"])
        self.fc2 = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=2)


class CNN(nn.Module):
    # Note: still need to modify so batch is 0th dimension

    def __init__(self, conf_dict):
        super(CNN, self).__init__()

        self.num_channels = 1 + int(conf_dict["deltas"]) + int(conf_dict["deltaDeltas"])
        self.num_features = conf_dict["num_features"]
        self.window_size = conf_dict["window_size"]
        self.kernel_size = conf_dict["kernel_size"]
        self.max_pool = conf_dict["max_pooling"]
        self.conv1 = nn.Conv2d(self.num_channels, conf_dict["num_feature_maps"], kernel_size=conf_dict["kernel_size"])

        width = torch.floor(torch.tensor((self.window_size - self.kernel_size[1] + 1)/self.max_pool[1])).to(int)
        height = torch.floor(torch.tensor((int(conf_dict["num_coeffs"]) + int(conf_dict["use_energy"]) - self.kernel_size[0] + 1)/self.max_pool[0])).to(int)
        self.fc1 = nn.Linear(width*height*conf_dict["num_feature_maps"], conf_dict["num_hidden"])
        self.fc2 = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])

    def forward(self, x):
        x = self.input_to_featmap(x)

        # Pass through network
        x = F.max_pool2d(torch.relu(self.conv1(x)), self.max_pool)
        x = x.view(x.size()[0], x.size()[1]*x.size()[2]*x.size()[3])
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def input_to_featmap(self, x):
        # Add zero padding in time
        x0 = torch.zeros((self.window_size-1, x.size()[1]), dtype=torch.float)
        if torch.cuda.is_available():
            x0 = x0.cuda()
        x = torch.cat((x0, x), dim=0)

        # Separate MFCCs and deltas
        x = torch.transpose(x, 0, 1)
        x = x.view(1, self.num_channels, int(self.num_features/self.num_channels), x.size()[1])

        # Format into feature maps
        batch_sz = x.size()[3] - self.window_size + 1
        idx = torch.linspace(0, self.window_size-1, self.window_size)
        idx = idx.repeat(batch_sz, 1) + torch.linspace(0, batch_sz-1, batch_sz).view(batch_sz, 1)
        idx = idx.to(int)
        x = x[0, :, :, idx]
        x = x.permute(2, 0, 1, 3)

        return x


class RNNModel(nn.Module):

    def __init__(self, conf_dict):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"], num_layers=1,
                          batch_first=True, bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_classes"])

    def forward(self, x):
        # Pass through recurrent layer
        h, _ = self.rnn(x)

        # Invert tanh and apply sigmoid
        h = 1/(1 + torch.sqrt((1 - h)/(1 + h)))

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=2)

        return out


class LSTMModel(nn.Module):

    def __init__(self, conf_dict):
        super(LSTMModel, self).__init__()

        # Stacked LSTMs
        if "num_layers" in conf_dict.keys():
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                num_layers=conf_dict["num_layers"], batch_first=True,
                                bidirectional=conf_dict["bidirectional"])
        # Default is one LSTM layer
        else:
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                batch_first=True, bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_classes"])

    def forward(self, x):
        # Pass through LSTM layer
        h, (_, _) = self.lstm(x)

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=2)

        return out


class GRUModel(nn.Module):

    def __init__(self, conf_dict):
        super(GRUModel, self).__init__()

        self.gru = nn.GRU(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"], num_layers=1,
                          batch_first=True, bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_classes"])

    def forward(self, x):
        # Pass through GRU layer
        h, _ = self.gru(x)

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=2)

        return out


class LSTMLM(nn.Module):

    def __init__(self, conf_dict):
        super(LSTMLM, self).__init__()

        # Used to fit initial phoneme distribution
        self.init_phoneme_counts = torch.zeros(1, conf_dict["num_classes"])
        self.init_phoneme_dist = torch.zeros(1, conf_dict["num_classes"])

        # Stacked LSTMs
        if "num_layers" in conf_dict.keys():
            self.lstm = nn.LSTM(input_size=conf_dict["num_classes"], hidden_size=conf_dict["num_hidden"],
                                num_layers=conf_dict["num_layers"], batch_first=True)
        # Default is one LSTM layer
        else:
            self.lstm = nn.LSTM(input_size=conf_dict["num_classes"], hidden_size=conf_dict["num_hidden"],
                                batch_first=True)

        self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])

    def init_state(self, batch_size, hidden_size):
        h0 = torch.zeros(1, batch_size, hidden_size)
        c0 = torch.zeros(1, batch_size, hidden_size)

        return h0, c0

    def forward(self, x, prev_state):
        # Pass through LSTM layer
        h, state = self.lstm(x, prev_state)

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=2)

        return out, state


class LSTMJoint(nn.Module):

    def __init__(self, conf_dict):
        super(LSTMJoint, self).__init__()

        # Stacked LSTMs
        if "num_layers" in conf_dict.keys():
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"]+conf_dict["num_classes"],
                                hidden_size=conf_dict["num_hidden"], num_layers=conf_dict["num_layers"],
                                batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"]+conf_dict["num_classes"],
                                hidden_size=conf_dict["num_hidden"], batch_first=True)

        self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_classes"])

    def init_state(self, batch_size, hidden_size):
        h0 = torch.zeros(1, batch_size, hidden_size)
        c0 = torch.zeros(1, batch_size, hidden_size)

        return h0, c0

    def forward(self, x, prev_state):
        # Pass through LSTM layer
        h, state = self.lstm(x, prev_state)

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc(h), dim=2)

        return out, state
    

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

    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight.data, a=a, b=b)
        nn.init.uniform_(m.bias.data, a=a, b=b)

    if isinstance(m, nn.RNN) or isinstance(m, nn.LSTM):
        nn.init.uniform_(m.weight_ih_l0, a=a, b=b)
        nn.init.uniform_(m.weight_hh_l0, a=a, b=b)
        nn.init.uniform_(m.bias_ih_l0, a=a, b=b)
        nn.init.uniform_(m.bias_hh_l0, a=a, b=b)


def initialize_network(conf_dict):
    """ Create network from configuration file and initialize
    weights

    Args:
        conf_dict (dict): conf file represented as a dict

    Returns:
        model (model): initialized model

    """
    # Get appropriate model type
    model = get_model_type(conf_dict)

    # Randomly initialize weights
    model.apply(initialize_weights)

    return model


def initialize_pretrained_network(conf_dict, pretrained_model):
    """ Create network from coniguration file and use
    pretrained weights

    Args:
        conf_dict (dict): conf file represented as a dict

    Returns:
        model (model): intialized model
    """
    # Get appropriate model type
    model = get_model_type(conf_dict)

    # For LSTM layer, use weights from pretrained model
    model.lstm = pretrained_model.lstm

    # Randomly initialize fully connected layer
    model.fc.apply(initialize_weights)

    return model


def get_model_type(conf_dict):

    if conf_dict["model_type"] == "SR":
        model = SoftmaxRegression(conf_dict)
    elif conf_dict["model_type"] == "MLP":
        model = MLP(conf_dict)
    elif conf_dict["model_type"] == "CNN":
        model = CNN(conf_dict)
    elif conf_dict["model_type"] == "RNN":
        model = RNNModel(conf_dict)
    elif conf_dict["model_type"] == "LSTM":
        model = LSTMModel(conf_dict)
    elif conf_dict["model_type"] == "GRU":
        model = GRUModel(conf_dict)

    return model
