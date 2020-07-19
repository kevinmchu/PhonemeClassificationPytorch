import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import os.path
import pickle
from pathlib import Path

# Features
from feature_extraction import fit_normalizer
from feature_extraction import read_feat_file
from sklearn import preprocessing

# Labels
from phone_mapping import get_phone_list
from phone_mapping import get_phoneme_list
from phone_mapping import get_moa_list
from phone_mapping import get_label_encoder

# Training and testing data
from validation import read_feat_list
from validation import train_val_split

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import initialize_network

from tqdm import tqdm
import logging
import re
from shutil import copyfile

from main import get_device
from main import train
from main import validate
from main import read_conf


def tune_hyperparameters(conf_file):
    """ Train and evaluate a phoneme classification model

    Args:
        model_type (str): model type
        train_list (list): list of training files
        valid_list (list): list of validation files
        label_type (str): phone or phoneme

    """
    # Read in conf file
    conf_dict = read_conf(conf_file)

    # Label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Random search over hyperparameters
    hyperparams = {}
    num_combos = 1
    hyperparams["learning_rate"] = 10.0**(np.random.randint(-5, -4, num_combos))
    hyperparams["momentum"] = 0.1*np.random.randint(9, 10, num_combos)
    #hyperparams["window_size"] = np.random.randint(10, 16, num_combos)
    hyperparams["num_layers"] = 1
    hyperparams["num_hidden"] = np.random.choice(np.array([116]), num_combos)
    hyperparams["acc"] = np.zeros((num_combos,))

    for i in range(num_combos):
        print(hyperparams)

        # Set current values of hyperparameters
        conf_dict["learning_rate"] = hyperparams["learning_rate"][i]
        conf_dict["momentum"] = hyperparams["momentum"][i]
        conf_dict["num_layers"] = hyperparams["num_layers"]
        conf_dict["num_hidden"] = hyperparams["num_hidden"][i]

        # Initialize network
        model = initialize_network(conf_dict)

        # Send network to GPU (if applicable)
        device = get_device()
        model.to(device)

        # Stochastic gradient descent with user-defined learning rate and momentum
        optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
        #optimizer = optim.Adam(model.parameters(), lr=conf_dict["learning_rate"])

        # Read in feature files
        train_list = read_feat_list(conf_dict["training"])
        valid_list = read_feat_list(conf_dict["development"])

        # Get standard scaler
        scaler = fit_normalizer(train_list, conf_dict)

        # Training
        logging.info("Training")
        max_acc = 0
        acc = []

        for epoch in tqdm(range(conf_dict["num_epochs"])):
            train(model, optimizer, le, conf_dict, train_list, scaler)
            valid_metrics = validate(model, le, conf_dict, valid_list, scaler)
            acc.append(valid_metrics["acc"])
            print("Validation Accuracy: {}".format(round(valid_metrics['acc'], 3)))

            # Track the best model
            if valid_metrics['acc'] > max_acc:
                max_acc = valid_metrics["acc"]

            # Stop early if accuracy does not improve over last 10 epochs
            if epoch >= 10:
                if acc[-1] - acc[-11] < 0.001:
                    break

        # Save the accuracy of the best model for current set of hyperparameters
        hyperparams["acc"][i] = max_acc

    # Set of hyperparameters that gives the highest accuracy on the validation set
    best_idx = np.argmax(hyperparams["acc"])
    best_hyperparams = {}
    best_hyperparams["learning_rate"] = hyperparams["learning_rate"][best_idx]
    best_hyperparams["momentum"] = hyperparams["momentum"][best_idx]
    best_hyperparams["num_hidden"] = hyperparams["num_hidden"][best_idx]

    print(best_hyperparams)


if __name__ == '__main__':
    # Necessary files
    conf_file = "conf/CNN_anechoic_mspec.txt"

    # Train and validate
    tune_hyperparameters(conf_file)
