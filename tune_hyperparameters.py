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

    for i in range(1):
        # Model directory
        model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[1]).replace(".txt", ""),
                                 "model" + str(i))
        # Path(model_dir).mkdir(parents=True, exist_ok=True)
        #
        # # Copy config file
        # copyfile(conf_file, (conf_file.replace("conf/", model_dir + "/")).replace(conf_file.split("/")[1], "conf.txt"))
        #
        # # Configure log file
        # logging.basicConfig(filename=model_dir + "/log", filemode="w", level=logging.INFO)

        # Instantiate the network
        # logging.info("Initializing model")
        model = initialize_network(conf_dict)

        # Send network to GPU (if applicable)
        device = get_device()
        model.to(device)

        # Stochastic gradient descent with user-defined learning rate and momentum
        optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])

        # Read in feature files
        train_list = read_feat_list(conf_dict["training"])
        valid_list = read_feat_list(conf_dict["development"])

        # Get standard scaler
        scale_file = model_dir + "/scaler.pickle"
        # scaler = fit_normalizer(train_list, conf_dict)
        # with open(scale_file, 'wb') as f:
        #     pickle.dump(scaler, f)

        # # Training curves
        # training_curves = model_dir + "/training_curves"
        # with open(training_curves, "w") as file_obj:
        #     file_obj.write("Epoch,Training Accuracy,Training Loss,Validation Accuracy,Validation Loss\n")

        # Training
        conf_dict["learning_rate"] = 1e-3
        logging.info("Training")
        for epoch in tqdm(range(conf_dict["num_epochs"])):
            # with open(training_curves, "a") as file_obj:
            # logging.info("Epoch: {}".format(epoch + 1))

            train(model, optimizer, le, conf_dict, train_list, scale_file)
            # train_metrics = validate(model, le, conf_dict, train_list, scale_file)
            valid_metrics = validate(model, le, conf_dict, valid_list, scale_file)
            print("Epoch: {}, Validation Accuracy: {}\n".format(epoch + 1, round(valid_metrics['acc'], 3)))

            # file_obj.write("{},{},{},{},{}\n".
            #                format(epoch + 1, round(train_metrics['acc'], 3), round(train_metrics['loss'], 3),
            #                       round(valid_metrics['acc'], 3), round(valid_metrics['loss'], 3)))

        # # Save model
        # torch.save(model, model_dir + "/model")


if __name__ == '__main__':
    # Necessary files
    conf_file = "conf/LSTM_rev_mfcc.txt"

    # Train and validate
    tune_hyperparameters(conf_file)