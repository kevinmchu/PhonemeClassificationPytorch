import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path
from datetime import datetime

# Features
from file_loader import collate_fn
from file_loader import Dataset
from file_loader import fit_normalizer
from file_loader import read_feat_file
from file_loader import TorchStandardScaler
from sklearn import preprocessing

# Labels
from phone_mapping import get_label_encoder
from phone_mapping import get_moa_list
from phone_mapping import phone_to_moa

# Training and testing data
from validation import read_feat_list

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import LSTMLM
from net import initialize_weights
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


def train(model, optimizer, conf_dict, train_generator):
    """ Train a phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        optimizer (optim.SGD): pytorch optimizer
        conf_dict (dict): configuration parameters
        train_generator
        
    Returns:
        none
        
    """
    # Training mode
    model.train()
    
    # Get device
    device = get_device()

    # Initial states
    h0, c0 = model.init_state(conf_dict["batch_size"], conf_dict["num_hidden"])
    h0 = h0.to(device)
    c0 = c0.to(device)

    for _, y_batch, loss_mask in train_generator:        
        optimizer.zero_grad()

        # Convert to tensor
        y_batch = torch.from_numpy(y_batch)

        # Get one-hot encoded vectors
        X_batch = (F.one_hot(y_batch, num_classes=conf_dict["num_classes"])).float()

        # Initial time step will be zeros
        X_batch = torch.cat((torch.zeros(conf_dict["batch_size"], 1, conf_dict["num_classes"]), X_batch), 1)

        # Move to GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        loss_mask = (torch.from_numpy(loss_mask)).to(device)

        # Get outputs
        train_outputs, (_, _) = model(X_batch[:, 0:-1, :], (h0, c0))
        train_outputs = train_outputs.permute(0, 2, 1)
    
        # Calculate loss
        # Note: Loss is set up so model tries to predict next phoneme in the sequence
        loss = torch.sum(loss_mask * F.nll_loss(train_outputs, y_batch, reduction='none'))/conf_dict["batch_size"]

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()


def validate(model, conf_dict, valid_generator):
    """ Validate phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        conf_dict (dict): configuration parameters
        valid_generator
        
    Returns:
        metrics (dict): loss and accuracy averaged across batches
    
    """

    metrics = {}

    # Running values
    num_frames = 0
    running_loss = 0

    # Get device
    device = get_device()

    # Evaluation mode
    model.eval()

    # Initial states
    h0, c0 = model.init_state(conf_dict["batch_size"], conf_dict["num_hidden"])
    h0 = h0.to(device)
    c0 = c0.to(device)

    with torch.no_grad():
        for _, y_val, loss_mask in valid_generator:
            # Convert to tensor
            y_val = torch.from_numpy(y_val)

            # Get one hot encoded vectors
            X_val = (F.one_hot(y_val, num_classes=conf_dict["num_classes"])).float()

            # Initial time step will be all zeros
            X_val = torch.cat((torch.zeros(conf_dict["batch_size"], 1, conf_dict["num_classes"]), X_val), 1)
            
            # Move to GPU
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            loss_mask = (torch.from_numpy(loss_mask)).to(device)

            # Get outputs
            outputs, (_, _) = model(X_val[:, 0:-1, :], (h0, c0))
            outputs = outputs.permute(0, 2, 1)

            # Initial states
            #h_prev, c_prev = model.init_state(conf_dict["batch_size"], conf_dict["num_hidden"])
            #y_prev = torch.zeros(conf_dict["batch_size"], 1, conf_dict["num_classes"])
            #h_prev = h_prev.to(device)
            #c_prev = c_prev.to(device)
            #y_prev = y_prev.to(device)
    
            # Get outputs and predictions for each time step
            #outputs = torch.zeros(conf_dict["batch_size"], loss_mask.size()[1], conf_dict["num_classes"], dtype=torch.float32)
            #for t in range(0, loss_mask.size()[1]):
            #    y_prev, (h_prev, c_prev) = model(y_prev, (h_prev, c_prev))
            #    y_prev = (F.one_hot(torch.argmax(y_prev, dim=2), num_classes=conf_dict["num_classes"])).float()
            #    outputs[:, t, :] = y_prev
            #outputs = outputs.permute(0, 2, 1)

            # Calculate running loss and running number of frames
            running_loss += (torch.sum(loss_mask * F.nll_loss(outputs, y_val, reduction='none'))).item()
            num_frames += (torch.sum(loss_mask)).item()

    # Average loss over all batches
    metrics['loss'] = running_loss / num_frames

    return metrics


def convert_string(key, value):
    """ Convert string into appropriate data type based on the
    dictionary key

    Args:
        key (str): dictionary key
        value (str): value expressed as a string

    Returns:
        converted_value (varies): value converted into appropriate data type

    """

    try:
        # Ints
        if "num" in key or "size" in key:
            converted_value = int(value)
        # Floats
        else:
            converted_value = float(value)
    except ValueError:
        # Tuple
        if re.match("\(\d*,\d*\)", value):
            temp = re.sub("\(|\)", "", value).split(",")
            converted_value = (int(temp[0]), int(temp[1]))
        # Boolean
        elif value == "True" or value == "False":
            converted_value = value == "True"
        # String
        else:
            converted_value = value

    return converted_value


def read_conf(conf_file):
    """ Read configuration file as dict

    Args:
        conf_file (str): configuration file

    Returns:
        conf_dict (dict): configuration file as dict

    """
    with open(conf_file, "r") as file_obj:
        conf = file_obj.readlines()

    conf = list(map(lambda x: x.replace("\n", ""), conf))

    # Convert conf to dict
    conf_dict = {}
    for line in conf:
        if "=" in line:
            contents = line.split(" = ")
            conf_dict[contents[0]] = convert_string(contents[0], contents[1])

    conf_dict["num_features"] = (1 + int(conf_dict["deltas"]) + int(conf_dict["deltaDeltas"])) * \
                                (conf_dict["num_coeffs"] + int(conf_dict["use_energy"]))

    return conf_dict


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
        model_dir = os.path.join("lm", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""), "model" + str(i))
        while os.path.exists(model_dir):
            i += 1
            model_dir = os.path.join("lm", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""),
                                     "model" + str(i))
        Path(model_dir).mkdir(parents=True)

        # Copy config file
        copyfile(conf_file, (conf_file.replace("conf/"+conf_dict["label_type"]+"/", model_dir + "/")).replace(conf_file.split("/")[2], "conf.txt"))

        # Configure log file
        logging.basicConfig(filename=model_dir+"/log", filemode="w", level=logging.INFO)

        ########## CREATE MODEL ##########
        # Instantiate the network
        logging.info("Initializing model")
        model = LSTMLM(conf_dict)
        model.apply(initialize_weights)
        model.to(get_device())

        # Configure optimizer
        if conf_dict["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
        elif conf_dict["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=conf_dict["learning_rate"])

        ########## TRAINING ##########
        # Training curves
        training_curves = model_dir + "/training_curves"
        with open(training_curves, "w") as file_obj:
            file_obj.write("Epoch,Validation Loss\n")

        logging.info("Training")

        # Generators
        train_set = Dataset(train_list, conf_dict, le)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=conf_dict["batch_size"],
                                                      num_workers=4, collate_fn=collate_fn, shuffle=True)
        valid_set = Dataset(valid_list, conf_dict, le)
        valid_generator = torch.utils.data.DataLoader(valid_set, batch_size=conf_dict["batch_size"],
                                                      num_workers=4, collate_fn=collate_fn, shuffle=True)

        # Used to track minimum loss
        min_loss = float("inf")
        loss = []

        iterator = tqdm(range(conf_dict["num_epochs"]))

        for epoch in iterator:
            with open(training_curves, "a") as file_obj:
                current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
                logging.info("Time: {}, Epoch: {}".format(current_time, epoch+1))

                # Train
                train(model, optimizer, conf_dict, train_generator)

                # Validate
                valid_metrics = validate(model, conf_dict, valid_generator)
                loss.append(valid_metrics["loss"])

                file_obj.write("{},{}\n".format(epoch+1, round(valid_metrics["loss"], 3)))

                # Track the best model and create checkpoint
                if valid_metrics['loss'] < min_loss:
                    min_loss = valid_metrics["loss"]
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                               model_dir + "/checkpoint.pt")

                # Stop early if accuracy does not improve over last 10 epochs
                if epoch >= 10:
                    if loss[-1] - loss[-11] >= 0:
                        logging.info("Detected minimum validation loss. Stopping early.")
                        iterator.close()
                        break


if __name__ == '__main__':
    # User inputs
    conf_file = "conf_lm/phoneme/LSTM_sim_rev_fftspec_ci.txt"
    num_models = 1

    # Train and validate model
    train_and_validate(conf_file, num_models)
