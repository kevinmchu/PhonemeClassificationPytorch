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

# Evaluation
from sklearn.metrics import confusion_matrix
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
from confusion_matrix import sort_classes
from confusion_matrix import plot_confusion_matrix
from confusion_matrix import plot_phoneme_confusion_matrix
from confusion_matrix import plot_moa_confusion_matrix
from plot_probs import plot_outputs

from tqdm import tqdm
import logging
import re
from shutil import copyfile

def get_device():
    """

    Returns:

    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


def train(model, optimizer, le, label_type, file_list, scale_file):
    """ Train a phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        optimizer (optim.SGD): pytorch optimizer
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        label_type (str): label type
        file_list (list): files in the test set
        
    Returns:
        none
        
    """
    # Shuffle
    random.shuffle(file_list)

    # Get device
    device = get_device()

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Set model to training mode
    model.train()

    losses = []

    for file in file_list:
        # Clear existing gradients
        optimizer.zero_grad()

        # Extract features and labels for current file
        x_batch, y_batch = read_feat_file(file, label_type)

        # Normalize features
        x_batch = scaler.fit_transform(x_batch)

        # Encode labels and integers
        y_batch = le.transform(y_batch).astype('long')

        # Move to GPU
        x_batch = (torch.from_numpy(x_batch)).to(device)
        y_batch = (torch.from_numpy(y_batch)).to(device)

        # Get outputs
        train_outputs = model(x_batch)

        # Calculate loss
        # Note: NLL loss actually calculates cross entropy loss when given values
        # transformed by log softmax
        loss = F.nll_loss(train_outputs, y_batch, reduction='sum')
        losses.append(loss.to('cpu').detach().numpy())

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()


def validate(model, le, label_type, file_list, scale_file):
    """ Validate phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        label_type (str): label type
        file_list (list): files in the test set
        
    Returns:
        avg_loss (float): loss averaged over all batches
    
    """

    metrics = {}

    # Running values
    running_correct = 0
    num_frames = 0
    running_loss = 0

    # Shuffle
    random.shuffle(file_list)

    # Get device
    device = get_device()

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Evaluation mode
    model.eval()

    with torch.no_grad():
        for file in file_list:
            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file, label_type)

            # Normalize features
            x_batch = scaler.fit_transform(x_batch)

            # Encode labels as integers
            y_batch = le.transform(y_batch).astype('long')

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)
            y_batch = (torch.from_numpy(y_batch)).to(device)

            # Get outputs and predictions
            outputs = model(x_batch)

            # Update running loss
            running_loss += (F.nll_loss(outputs, y_batch, reduction='sum')).item()

            # Calculate accuracy
            matches = y_batch.cpu().numpy() == torch.argmax(outputs, dim=1).cpu().numpy()
            running_correct += np.sum(matches)
            num_frames += len(matches)

    # Average loss over all batches
    metrics['acc'] = running_correct / num_frames
    metrics['loss'] = running_loss / num_frames

    return metrics


def read_conf(conf_file):
    with open(conf_file, "r") as file_obj:
        conf = file_obj.readlines()

    conf = list(map(lambda x: x.replace("\n", ""), conf))

    # Convert conf to dict
    conf_dict = {}
    for line in conf:
        if "=" in line:
            contents = line.split(" = ")
            try:
                # Ints
                if "num" in contents[0]:
                    conf_dict[contents[0]] = int(contents[1])
                # Floats
                else:
                    conf_dict[contents[0]] = float(contents[1])
            except ValueError:
                # Tuple
                if re.match("\(\d*,\d*\)", contents[1]):
                    temp = re.sub("\(|\)", "", contents[1]).split(",")
                    conf_dict[contents[0]] = (int(temp[0]), int(temp[1]))
                # Boolean
                elif contents[1] == "True" or contents[1] == "False":
                    conf_dict[contents[0]] = contents[1] == "True"
                # String
                else:
                    conf_dict[contents[0]] = contents[1]

    return conf_dict


def train_and_validate(conf_file):
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
        model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[1]).replace(".txt", ""), "model" + str(i))
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Copy config file
        copyfile(conf_file, (conf_file.replace("conf/", model_dir + "/")).replace(conf_file.split("/")[1], "conf.txt"))

        # Configure log file
        logging.basicConfig(filename=model_dir+"/log", filemode="w", level=logging.INFO)

        # Instantiate the network
        logging.info("Initializing model")
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
        scaler = fit_normalizer(train_list, conf_dict["label_type"])
        with open(scale_file, 'wb') as f:
            pickle.dump(scaler, f)

        # Training curves
        training_curves = model_dir + "/training_curves"
        with open(training_curves, "w") as file_obj:
            file_obj.write("Epoch,Training Accuracy,Training Loss,Validation Accuracy,Validation Loss\n")

        # Training
        logging.info("Training")
        for epoch in tqdm(range(conf_dict["num_epochs"])):
            with open(training_curves, "a") as file_obj:
                logging.info("Epoch: {}".format(epoch+1))

                train(model, optimizer, le, conf_dict["label_type"], train_list, scale_file)
                train_metrics = validate(model, le, conf_dict["label_type"], train_list, scale_file)
                valid_metrics = validate(model, le, conf_dict["label_type"], valid_list, scale_file)

                file_obj.write("{},{},{},{},{}\n".
                                format(epoch+1, round(train_metrics['acc'], 3), round(train_metrics['loss'], 3),
                                        round(valid_metrics['acc'], 3), round(valid_metrics['loss'], 3)))

        # Save model
        torch.save(model, model_dir + "/model")


if __name__ == '__main__':
    # Necessary files
    conf_file = "conf/LSTM_anechoic_mfcc.txt"
    test_feat_list = "data/test_anechoic/mfcc.txt"

    # Train and validate
    train_and_validate(conf_file)
