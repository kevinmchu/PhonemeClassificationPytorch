# plot_metrics.py
# Author: Kevin Chu
# Last Modified: 05/20/2020

import re
import matplotlib.pyplot as plt
from matplotlib import style


def read_metrics(filename):
    """ Read metrics in from file

    This function reads accuracy and loss metrics as a function
    of epoch.

    Args:
        filename (str): name of log file

    Returns:
        metrics (dict): dictionary with loss and accuracy
    """

    metrics = {'epoch': [], 'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

    # Read in file
    with open(filename) as f:
        for i, line in enumerate(f):
            # Skip first line since just contains headings
            if i > 0:
                if re.match("INFO:root:", line) is not None:
                    line = line.replace("INFO:root:", "").split(",")
                    metrics['epoch'].append(float(line[0]))
                    metrics['train_acc'].append(float(line[1])*100)
                    metrics['train_loss'].append(float(line[2]))
                    metrics['val_acc'].append(float(line[3])*100)
                    metrics['val_loss'].append(float(line[4]))

    return metrics


def plot_acc_and_loss(filename):
    """ Plots accuracy and loss

    This function plots the accuracy and the loss of the training
    and validation data.

    Args:
        filename (str): name of log file

    Returns:
        none
    """
    metrics = read_metrics(filename)

    # Plot the figure
    style.use("ggplot")
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1), sharex=ax1)

    ax1.plot(metrics['epoch'], metrics['train_acc'], label="Training")
    ax1.plot(metrics['epoch'], metrics['val_acc'], label="Validation")
    ax1.set_yticks([35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    ax1.set_yticklabels([35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    ax1.set_ylim([35, 85])
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("% Correct Frames")
    ax1.legend(loc=2)

    ax2.plot(metrics['epoch'], metrics['train_loss'], label="Training")
    ax2.plot(metrics['epoch'], metrics['val_loss'], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Cross Entropy")
    ax2.legend(loc=2)

    plt.show()


def plot_losses(filenames):

    style.use("ggplot")

    for filename in filenames:
        metrics = read_metrics(filename)
        plt.plot(metrics['epoch'], metrics['train_acc'])

    plt.xlim([0, 400])
    plt.ylim([35, 85])
    plt.yticks([35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    #plt.yticklabels([35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85])
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("% Correct Frames")
    plt.legend(["MLP", "RNN", "BRNN", "LSTM", "BLSTM"])
    plt.show()


if __name__ == '__main__':
    # filename = "log/MLP_61phones"
    # plot_acc_and_loss(filename)

    filenames = ["log/MLP_61phones", "log/RNN_61phones_sigmoid", "log/BRNN_61phones_sigmoid_norm",
                 "log/LSTM_61phones_norm", "log/BLSTM_61phones_norm"]

    plot_losses(filenames)
