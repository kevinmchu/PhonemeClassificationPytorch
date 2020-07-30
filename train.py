import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path

# Features
from feature_extraction import fit_normalizer
from feature_extraction import read_feat_file

# Labels
from phone_mapping import get_label_encoder

# Training and testing data
from validation import read_feat_list

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

def get_device():
    """

    Returns:

    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


def train(model, optimizer, le, conf_dict, file_list, scaler):
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
    # Shuffle
    random.shuffle(file_list)

    # Get device
    device = get_device()

    # Set model to training mode
    model.train()

    losses = []

    for file in file_list:
        # Clear existing gradients
        optimizer.zero_grad()

        # Extract features and labels for current file
        x_batch, y_batch = read_feat_file(file, conf_dict)

        # Normalize features
        x_batch = scaler.transform(x_batch)

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


def validate(model, le, conf_dict, file_list, scaler):
    """ Validate phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        conf_dict (dict): configuration parameters
        file_list (list): files in the test set
        scaler (StandardScaler): scales features to zero mean unit variance
        
    Returns:
        metrics (dict): loss and accuracy averaged across batches
    
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

    # Evaluation mode
    model.eval()

    with torch.no_grad():
        for file in file_list:
            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file, conf_dict)

            # Normalize features
            x_batch = scaler.transform(x_batch)

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

    # Calculate number of features
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

    # Label encoder
    le = get_label_encoder(conf_dict["label_type"])

    for i in range(num_models):
        # Model directory - create new folder for each new instance of a model
        model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[1]).replace(".txt", ""), "model" + str(i))
        while os.path.exists(model_dir):
            i += 1
            model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[1]).replace(".txt", ""),
                                     "model" + str(i))
        Path(model_dir).mkdir(parents=True)

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
        scaler = fit_normalizer(train_list, conf_dict)
        with open(scale_file, 'wb') as f:
            pickle.dump(scaler, f)

        # Training curves
        training_curves = model_dir + "/training_curves"
        with open(training_curves, "w") as file_obj:
            file_obj.write("Epoch,Training Accuracy,Training Loss,Validation Accuracy,Validation Loss\n")

        # Training
        logging.info("Training")
        max_acc = 0
        acc = []

        for epoch in tqdm(range(conf_dict["num_epochs"])):
            with open(training_curves, "a") as file_obj:
                logging.info("Epoch: {}".format(epoch+1))

                train(model, optimizer, le, conf_dict, train_list, scaler)
                train_metrics = validate(model, le, conf_dict, train_list, scaler)
                valid_metrics = validate(model, le, conf_dict, valid_list, scaler)
                acc.append(valid_metrics["acc"])

                file_obj.write("{},{},{},{},{}\n".
                                format(epoch+1, round(train_metrics['acc'], 3), round(train_metrics['loss'], 3),
                                        round(valid_metrics['acc'], 3), round(valid_metrics['loss'], 3)))

                # Track the best model
                if valid_metrics['acc'] > max_acc:
                    max_acc = valid_metrics["acc"]
                    torch.save(model, model_dir + "/model")

                # Stop early if accuracy does not improve over last 10 epochs
                if epoch >= 10:
                    if acc[-1] - acc[-11] < 0.001:
                        logging.info("Detected maximum validation accuracy. Stopping early.")
                        break


if __name__ == '__main__':
    # User inputs
    conf_file = "conf/LSTM_rev_mspec.txt"
    num_models = 1

    # Train and validate
    train_and_validate(conf_file, num_models)
