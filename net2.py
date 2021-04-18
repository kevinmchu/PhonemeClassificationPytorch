import numpy
import torch
import torch.nn.functional as F


class MLP:

    def __init__(self, conf_dict):
        super(MLP, self).__init__()

        self.num_features = conf_dict["num_features"]
        self.num_hidden = conf_dict["num_hidden"]
        self.num_classes = conf_dict["num_classes"]

        # Initialize weights
        self.Wih = initialize_weights(self.num_hidden, self.num_features)
        self.bih = initialize_weights(self.num_hidden, 1)
        self.Who = initialize_weights(self.num_classes, self.num_hidden)
        self.bho = initialize_weights(self.num_classes, 1)

        # Initialize gradients
        self.dLdWih = torch.zeros(self.num_hidden, self.num_features)
        self.dLdbih = torch.zeros(self.num_hidden, 1)
        self.dLdWho = torch.zeros(self.num_classes, self.num_hidden)
        self.dLdbho = torch.zeros(self.num_classes, 1)

    def forward(self, x):
        # Transpose to (num_feats, num_timesteps)
        x = torch.transpose(x, 0, 1)

        # Matrices of ones to use for bias terms
        ones = torch.ones(1, x.size()[1])

        # Forward pass
        h = torch.sigmoid(self.Wih*x + self.bih*ones)
        yhat = F.log_softmax(self.Who*h + self.bho*ones, dim=1)

        return yhat

    def backward(self, x, h, y, yhat):
        """

        Args:
            x: (num_timesteps, num_features)
            h: (num_hidden, num_timesteps)
            y: (num_classes, num_timesteps)
            yhat: (num_classes, num_timesteps)

        Returns:

        """
        # Transpose x to (num_features, num_timesteps)
        x = torch.transpose(x, 0, 1)
        num_timesteps = x.size()[1]

        # Gradient of loss wrt input to softmax
        dLds = y*(yhat-1)

        # Gradient calculations
        for t in range(num_timesteps):
            # Hidden to output
            self.dLdWho += torch.ger(dLds[:, t].view(self.num_classes), h[:, t].view(self.num_hidden))
            self.dLdbho += dLds[:, t]

            # Gradient of loss wrt input to sigmoid
            dLdsig = torch.matmul(torch.transpose(self.Who, 0, 1), dLds[:, t]) * (h[:, t] * (1 - h[:, t]))

            # Input to hidden
            self.dLdWih += torch.ger(dLdsig.view(self.num_hidden), x[:, t].view(self.num_features))
            self.dLdbih += dLdsig


class RNN:

    def __init__(self, conf_dict):
        super(RNN, self).__init__()

        self.num_features = conf_dict["num_features"]
        self.num_hidden = conf_dict["num_hidden"]
        self.num_classes = conf_dict["num_classes"]

        # Initialize weights
        self.Wih = initialize_weights(self.num_hidden, self.num_features)
        self.bih = initialize_weights(self.num_hidden, 1)
        self.Whh = initialize_weights(self.num_hidden, self.num_hidden)
        self.bhh = initialize_weights(self.num_hidden, 1)
        self.Who = initialize_weights(self.num_classes, self.num_hidden)
        self.bho = initialize_weights(self.num_classes, 1)

        # Initialize gradients
        self.dLdWih = torch.zeros(self.num_hidden, self.num_features)
        self.dLdbih = torch.zeros(self.num_hidden, 1)
        self.dLdWhh = torch.zeros(self.num_hidden, self.num_hidden)
        self.dLdbhh = torch.zeros(self.num_hidden, 1)
        self.dLdWho = torch.zeros(self.num_classes, self.num_hidden)
        self.dLdbho = torch.zeros(self.num_classes, 1)


    def forward(self, x):
        # Transpose to (num_feats, num_timesteps)
        x = torch.transpose(x, 0, 1)
        num_timesteps = x.size()[1]

        # Initialize hidden state
        h0 = torch.zeros(self.num_hidden, 1)

        # Create matrix for hidden state and output
        h = torch.zeros(self.num_hidden, num_timesteps)

        for t in range(num_timesteps):
            if t == 0:
                h[:, t] = torch.tanh(self.Wih*x[:, t] + self.bih + self.Whh*h0 + self.bhh)
            else:
                h[:, t] = torch.tanh(self.Wih*x[:, t] + self.bih + self.Whh*h[:, t-1] + self.bhh)

        # Output
        yhat = F.log_softmax(self.Who*h, dim=1)

        return yhat


    def backward(self, x, h, y, yhat):
        """

        Args:
            x: (num_timesteps, num_features)
            h: (num_hidden, num_timesteps)
            y: (num_classes, num_timesteps), one hot encoded
            yhat: (num_classes, num_timesteps)

        Returns:

        """
        # Tranpose x to (num_features, num_timesteps)
        x = torch.transpose(x, 0, 1)
        num_timesteps = x.size()[1]

        # Gradient of loss wrt input to softmax
        dLds = y*(yhat-1)

        for t in range(0, num_timesteps)[::-1]:
            # Hidden to output
            self.dLdWho += torch.ger(dLds[:, t].view(self.num_classes), h[:, t].view(self.num_hidden))
            self.dLdbho += dLds[:, t]

            # Gradient of loss wrt hidden layer
            dLdh = torch.matmul(torch.transpose(self.Who, 0, 1), dLds[:, t]) * (1-h[:, t]**2)

            for i in range(0, t)[::-1]:
                self.dLdWhh += dLdh * h[:, i-1]
                self.dLdWih += dLdh

                dLdh = torch.matmul(torch.transpose(self.Whh, 0, 1), dLdh) * (1-h[:, i-1]**2)

        # Hidden to hidden and input to hidden
        for i in range(0, yhat.size()[1]):
            yhat_curr = yhat[y[:, i].nonzero(), i]
            dLdo = yhat_curr - 1
            self.dLdWho += dLdo * self.h[:, i]
            delta = self.dLdWho * self.w_ho * (1-self.h0**2)

            for j in range(1, i):
                self.dLdWhh += delta * self.h[:, i-1]
                self.dLdWih += delta
                delta *= self.Whh * (1-self.h[:, i]**2)


class LSTM:

    def __init__(self, conf_dict):
        super(LSTM, self).__init__()

        self.num_features = conf_dict["num_features"]
        self.num_hidden = conf_dict["num_hidden"]
        self.num_classes = conf_dict["num_classes"]

        # Initialize weights
        # Input gate
        self.Wii = initialize_weights(self.num_hidden, self.num_features)
        self.bii = initialize_weights(self.num_hidden, 1)
        self.Whi = initialize_weights(self.num_hidden, self.num_hidden)
        self.bhi = initialize_weights(self.num_hidden, 1)

        # Forget gate
        self.Wif = initialize_weights(self.num_hidden, self.num_features)
        self.bif = initialize_weights(self.num_hidden, 1)
        self.Whf = initialize_weights(self.num_hidden, self.num_hidden)
        self.bhf = initialize_weights(self.num_hidden, 1)

        # Cell gate
        self.Wig = initialize_weights(self.num_hidden, self.num_features)
        self.big = initialize_weights(self.num_hidden, 1)
        self.Whg = initialize_weights(self.num_hidden, self.num_hidden)
        self.bhg = initialize_weights(self.num_hidden, 1)

        # Output gate
        self.Wio = initialize_weights(self.num_hidden, self.num_features)
        self.bio = initialize_weights(self.num_hidden, 1)
        self.Who = initialize_weights(self.num_hidden, self.num_hidden)
        self.bho = initialize_weights(self.num_hidden, 1)

    def forward(self, x):
        """

        Args:
            x: num_feats x num_timesteps

        Returns:

        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_hidden, 1)
        c0 = torch.zeros(self.num_hidden, 1)

        # Create matrices for hidden and cell states
        ht = torch.zeros(self.num_hidden, x.size()[1])
        ct = torch.zeros(self.num_hidden, x.size()[1])

        for t in range(x.size()[1]):
            if t == 0:
                h = h0
                c = c0
            else:
                h = ht[:, t-1]
                c = ct[:, t-1]

            # Gates
            it = torch.sigmoid(self.Wii*x[:, t] + self.bii + self.Whi*h + self.bhi)
            ft = torch.sigmoid(self.Wif*x[:, t] + self.bif + self.Whf*h + self.bhf)
            ot = torch.sigmoid(self.Wio*x[:, t] + self.bio + self.Who*h + self.bho)

            # Cell gate
            gt = torch.tanh(self.Wig*x[:, t] + self.big + self.Whg*h + self.bhg)

            # Cell state
            ct[:, t] = ft*c + it*gt

            # Hidden state
            ht[:, t] = ot*torch.tanh(ct)


def initialize_weights(out_dim, in_dim):
    a = -0.1
    b = 0.1
    w0 = (b-a) * torch.rand(out_dim, in_dim) + a

    return w0
