import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path
from datetime import datetime

# Features
from feature_extraction import collate_fn
from feature_extraction import Dataset
from feature_extraction import fit_normalizer
from feature_extraction import read_feat_file
from feature_extraction import TorchStandardScaler
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

    # Used to select random subset of training data - DELETE LATER
    idx = 0
    
    # Get device
    device = get_device()

    for X_batch, y_batch in train_generator:
        # Used to iterate through only a subset of data - DELETE LATER
        if idx >= np.round(conf_dict["train_subset"] * 3696):
            break
        else:
            idx += 1
        
        optimizer.zero_grad()

        # Move to GPU
        X_batch = torch_scaler.transform((torch.from_numpy(X_batch)).to(device))
        y_batch = (torch.from_numpy(y_batch)).to(device)

        # Get outputs
        train_outputs = model(X_batch)
        train_outputs = train_outputs.permute(0, 2, 1)

        # Loss mask - only calculate loss over valid region of the utterance (i.e. not inf padded)
        loss_mask = torch.prod(X_batch != np.inf, axis=2)
    
        # Calculate loss
        # Note: NLL loss actually calculates cross entropy loss when given values
        # transformed by log softmax
        loss = torch.sum(loss_mask * F.nll_loss(train_outputs, y_batch, reduction='none'))

        # Backpropagate
        loss.backward()

        # Gradient clipping, if applicable
        #if "gradient_clip_value" in conf_dict.keys():
        #    nn.utils.clip_grad_value_(model.parameters(), conf_dict["gradient_clip_value"])

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
        for X_val, y_val in valid_generator:
            # Move to GPU
            X_val = torch_scaler.transform((torch.from_numpy(X_val)).to(device))
            y_val = (torch.from_numpy(y_val)).to(device)

            # Get outputs and predictions
            outputs = model(X_val)
            outputs = outputs.permute(0, 2, 1)

            # Loss mask - only calculate loss over valid region of the utterance (i.e. not inf padded)
            loss_mask = torch.prod(X_val != np.inf, axis=2)
            running_loss += (torch.sum(loss_mask * F.nll_loss(outputs, y_val, reduction='none'))).item()

            # Calculate accuracy
            matches = (y_val == torch.argmax(outputs, dim=1))
            running_correct += (torch.sum(matches)).item()

            num_frames += (torch.sum(loss_mask)).item()

    # Average loss over all batches
    metrics['acc'] = running_correct / num_frames
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
        # scaler = fit_normalizer(train_list, conf_dict)
        with open(model_dir.replace("model" + str(i), "librispeech") + "/scaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
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
    conf_file = "conf/moa/LSTM_sim_rev_fftspec_ci.txt"
    num_models = 1

    # Train and validate model
    train_and_validate(conf_file, num_models)
