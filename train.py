import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path
from datetime import datetime

# Configuration file
from conf import read_conf

# Features
from file_loader import collate_fn
from file_loader import Dataset
from file_loader import fit_normalizer
from file_loader import read_feat_file
from file_loader import TorchStandardScaler
from sklearn import preprocessing

# Labels
from phone_mapping import get_label_encoder

# Training and testing data
from file_loader import read_feat_list

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import initialize_network
from net import get_model_type

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


def train(model, optimizer, le, conf_dict, train_generator, torch_scaler):
    """ Train a phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        optimizer (optim.SGD): pytorch optimizer
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        conf_dict (dict): configuration parameters
        file_list (list): files in the test set
        scaler (StandardScaler): scales features to zero mean unit variance
        
    Returns:
        none
        
    """
    # Training mode
    model.train()
    
    # Get device
    device = get_device()

    for X_batch, y_batch, loss_mask in train_generator:        
        optimizer.zero_grad()

        # Move to GPU
        X_batch = torch_scaler.transform((torch.from_numpy(X_batch)).to(device))
        y_batch = (torch.from_numpy(y_batch)).to(device)
        loss_mask = (torch.from_numpy(loss_mask)).to(device)

        # Get outputs
        train_outputs = model(X_batch)
        train_outputs = train_outputs.permute(0, 2, 1)
    
        # Calculate loss
        # Note: NLL loss actually calculates cross entropy loss when given values
        # transformed by log softmax
        loss = torch.sum(loss_mask * F.nll_loss(train_outputs, y_batch, reduction='none'))/X_batch.size()[0]

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()


def validate(model, le, conf_dict, valid_generator, torch_scaler):
    """ Validate phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        conf_dict (dict): configuration parameters
        file_list (list): files in the test set
        scaler (StandardScaler): scales features to zero mean unit variance
        moa_model (torch.nn.Module): moa model if doing hierarchical classification, otherwise empty list
        
    Returns:
        metrics (dict): loss and accuracy averaged across batches
    
    """

    metrics = {}

    # Running values
    running_correct = 0
    num_frames = 0
    running_loss = 0

    # Get device
    device = get_device()

    # Evaluation mode
    model.eval()

    with torch.no_grad():
        for X_val, y_val, loss_mask in valid_generator:
            # Move to GPU
            X_val = torch_scaler.transform((torch.from_numpy(X_val)).to(device))
            y_val = (torch.from_numpy(y_val)).to(device)
            loss_mask = (torch.from_numpy(loss_mask)).to(device)

            # Get outputs and predictions
            outputs = model(X_val)
            outputs = outputs.permute(0, 2, 1)

            # Loss mask - only calculate loss over valid region of the utterance (i.e. not padded)
            running_loss += (torch.sum(loss_mask * F.nll_loss(outputs, y_val, reduction='none'))).item()

            # Calculate accuracy
            matches = (y_val == torch.argmax(outputs, dim=1))
            running_correct += (torch.sum(loss_mask * matches)).item()

            num_frames += (torch.sum(loss_mask)).item()

    # Average loss over all batches
    metrics['acc'] = running_correct / num_frames
    metrics['loss'] = running_loss / num_frames

    return metrics


def train_and_validate(conf_file, num_models):
    """ Train and evaluate a phoneme classification model

    Args:
        conf_file (str): txt file containing model info
        num_models (int): number of instances to of model to train

    Returns
        none

    """
    # Read in conf file
    conf_dict = read_conf(conf_file)

    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])
    valid_list = read_feat_list(conf_dict["development"])

    # Label encoder
    le = get_label_encoder(conf_dict["label_type"])

    for i in range(num_models):
        # Model directory - create new folder for each new instance of a model
        model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""), "model" + str(i))
        while os.path.exists(model_dir):
            i += 1
            model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""),
                                     "model" + str(i))
        Path(model_dir).mkdir(parents=True)

        # Copy config file
        copyfile(conf_file, (conf_file.replace("conf/"+conf_dict["label_type"]+"/", model_dir + "/")).replace(conf_file.split("/")[2], "conf.txt"))

        # Configure log file
        logging.basicConfig(filename=model_dir+"/log", filemode="w", level=logging.INFO)

        # Get standard scaler
        device = get_device()
        scaler = fit_normalizer(train_list, conf_dict)
        #with open(model_dir.replace("model" + str(i), "librispeech") + "/scaler.pickle", 'rb') as f:
        #    scaler = pickle.load(f)
        with open(model_dir + "/scaler.pickle", 'wb') as f2:
            pickle.dump(scaler, f2)
        torch_scaler = TorchStandardScaler(scaler.mean_, scaler.var_, device)

        ########## CREATE MODEL ##########
        # Instantiate the network
        logging.info("Initializing model")
        model = initialize_network(conf_dict)
        
        if "pretrained_model_dir" in conf_dict.keys():
            pretrained_model_dir = os.path.join(conf_dict["pretrained_model_dir"], "model" + str(i))
            pretrained_dict = torch.load(pretrained_model_dir + "/model.pt")
            for i, param in enumerate(pretrained_dict):
                if i < len(pretrained_dict.keys())-2:
                    model.load_state_dict({param: pretrained_dict[param]}, strict=False)

        model.to(device)

        # Configure optimizer
        if conf_dict["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
        elif conf_dict["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=conf_dict["learning_rate"])

        ########## TRAINING ##########
        # Training curves
        training_curves = model_dir + "/training_curves"
        with open(training_curves, "w") as file_obj:
            file_obj.write("Epoch,Validation Accuracy,Validation Loss\n")

        logging.info("Training")

        # Generators
        train_set = Dataset(train_list, conf_dict, le)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=conf_dict["batch_size"],
                                                      num_workers=4, collate_fn=collate_fn, shuffle=True)
        valid_set = Dataset(valid_list, conf_dict, le)
        valid_generator = torch.utils.data.DataLoader(valid_set, batch_size=conf_dict["batch_size"],
                                                      num_workers=4, collate_fn=collate_fn, shuffle=True)

        # Used to track maximum accuracy
        max_acc = 0
        acc = []

        iterator = tqdm(range(conf_dict["num_epochs"]))

        for epoch in iterator:
            with open(training_curves, "a") as file_obj:
                current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
                logging.info("Time: {}, Epoch: {}".format(current_time, epoch+1))

                # Train
                train(model, optimizer, le, conf_dict, train_generator, torch_scaler)

                # Validate
                valid_metrics = validate(model, le, conf_dict, valid_generator, torch_scaler)
                acc.append(valid_metrics["acc"])

                file_obj.write("{},{},{}\n".
                                format(epoch+1, round(valid_metrics['acc'], 3), round(valid_metrics['loss'], 3)))

                # Track the best model and create checkpoint
                if valid_metrics['acc'] > max_acc:
                    max_acc = valid_metrics["acc"]
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                               model_dir + "/checkpoint.pt")

                # Stop early if accuracy does not improve over last 10 epochs
                if epoch >= 10:
                    if acc[-1] - acc[-11] < 0.001:
                        logging.info("Detected maximum validation accuracy. Stopping early.")
                        iterator.close()
                        break


if __name__ == '__main__':
    # User inputs
    conf_file = "conf/phoneme/LSTM_but_rev_fftspec_ci.txt"
    num_models = 1

    # Train and validate model
    train_and_validate(conf_file, num_models)
