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
from file_loader import fit_normalizer
from file_loader import read_feat_file
from sklearn import preprocessing

# Labels
from phone_mapping import get_label_encoder
from phone_mapping import get_phone_list
from phone_mapping import get_moa_list
from phone_mapping import phone_to_moa
from phone_mapping import phone_to_bpg

# Training and testing data
from file_loader import read_feat_list
import evaluate

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import initialize_network
from net import initialize_pretrained_network

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


def train(experts, optimizers, le, conf_dict, file_list, scaler):
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

    # Select random subset of training data if applicable
    if 'train_subset' in conf_dict.keys():
        file_list = file_list[0:round(conf_dict['train_subset'] * len(file_list))]

    # Get device
    device = get_device()

    for bpg in experts.keys():
        # Set model to training mode
        experts[bpg].train()

    for file in file_list:
        # Extract features and labels for current file
        x_batch, y_batch = read_feat_file(file, conf_dict)

        # Normalize features
        x_batch = scaler.transform(x_batch)

        # Convert phone labels to bpg
        if conf_dict["bpg"] == "moa":
            y_bpg = phone_to_moa(list(y_batch))
        elif conf_dict["bpg"] == "bpg":
            y_bpg = phone_to_bpg(list(y_batch))

        y_bpg = np.array(y_bpg)

        # Move normalized features to GPU
        x_batch = (torch.from_numpy(x_batch)).to(device)

        for bpg in experts.keys():
            # Clear existing gradients
            optimizers[bpg].zero_grad()

            bpg_idx = np.argwhere(y_bpg == bpg)
            bpg_idx = np.reshape(bpg_idx, (len(bpg_idx),))

            if len(bpg_idx) > 0:
                # Initialize y as array of -1
                # Used to replace irrelevant bpg's with -1
                y = -np.ones((len(y_batch, ))).astype('long')

                # Replace -1's at indices where phone is correct moa
                y[bpg_idx] = le[bpg].transform(np.array(y_batch)[bpg_idx]).astype('long')

                # Move to GPU
                y = (torch.from_numpy(y)).to(device)

                # Get outputs
                train_outputs = experts[bpg](x_batch)

                # Calculate loss
                # Ignore inputs that do not correspond to current moa
                loss = F.nll_loss(train_outputs, y, reduction='sum', ignore_index=-1)

                # Backpropagate and update weights
                loss.backward()
                optimizers[bpg].step()


def validate(experts, le, conf_dict, file_list, scaler, bpg_model):
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

    # Shuffle
    random.shuffle(file_list)

    # Get label encoder for phone/phoneme
    le_phone = get_label_encoder(conf_dict["label_type"])

    # If hierarchical classification, get label encoder for bpg
    if conf_dict["hierarchical"]:
        le_bpg = get_label_encoder(conf_dict["bpg"])

    # Get device
    device = get_device()

    # Evaluation mode
    bpg_model.eval()
    for bpg in experts.keys():
        experts[bpg].eval()

    with torch.no_grad():
        for file in file_list:
            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file, conf_dict)

            # Normalize features
            x_batch = scaler.transform(x_batch)

            # Encode labels as integers
            y_batch = le_phone.transform(y_batch).astype('long')

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)
            y_batch = (torch.from_numpy(y_batch)).to(device)

            # Get moa model outputs
            bpg_outputs = bpg_model(x_batch)

            # Get posterior probabilities from each expert
            posteriors = torch.zeros((len(y_batch), len(le_phone.classes_)))
            posteriors = posteriors.to(device)
            for bpg in experts.keys():
                outputs = experts[bpg](x_batch)
                posteriors[:, le_phone.transform(le[bpg].classes_)] = bpg_outputs[:, le_bpg.transform([bpg])] + outputs

            # Update running loss
            running_loss += (F.nll_loss(posteriors, y_batch, reduction='sum'))

            # Calculate accuracy
            matches = (y_batch == torch.argmax(posteriors, dim=1))
            running_correct += torch.sum(matches)
            num_frames += len(matches)

    # Average loss over all batches
    metrics['acc'] = running_correct.item() / num_frames
    metrics['loss'] = running_loss.item() / num_frames

    return metrics


def train_and_validate(conf_file, num_models):
    """ Train and evaluate a phoneme classification model

    Args:
        conf_file (str): txt file containing model info
        num_models (int): number of instances of model to train

    Returns
        none

    """
    # Read in conf file
    conf_dict = read_conf(conf_file)

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

        # Initializing the experts using same architecture
        logging.info("Initializing experts")
        experts = {}
        optimizers = {}
        le = {}
        device = get_device()

        num_classes = conf_dict["num_classes"]

        phone_list = get_phone_list()

        if conf_dict["bpg"] == "moa":
            phone_list_as_bpg = phone_to_moa(phone_list)
        elif conf_dict["bpg"] == "bpg":
            phone_list_as_bpg = phone_to_bpg(phone_list)

        le_bpg = get_label_encoder(conf_dict["bpg"])

        for bpg in le_bpg.classes_:
            # Initialize experts
            idx = np.argwhere(np.array(phone_list_as_bpg) == bpg)
            idx = np.reshape(idx, (len(idx),))
            conf_dict["num_classes"] = len(idx)
            experts[bpg] = initialize_network(conf_dict)

            experts[bpg].to(device)

            # Get label encoder
            le[bpg] = preprocessing.LabelEncoder()
            le[bpg].fit(list(np.array(phone_list)[idx]))

            # Different SGD optimizer for each expert
            optimizers[bpg] = optim.SGD(experts[bpg].parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])

        # Reset num_classes
        conf_dict["num_classes"] = num_classes

        # Read in feature files
        train_list = read_feat_list(conf_dict["training"])
        valid_list = read_feat_list(conf_dict["development"])

        # Get standard scaler
        scale_file = model_dir + "/scaler.pickle"
        scaler = fit_normalizer(train_list, conf_dict)
        with open(scale_file, 'wb') as f:
            pickle.dump(scaler, f)

        # Load trained moa model
        bpg_model_dir = os.path.join(conf_dict["bpg_model_dir"], "model" + str(i))
        bpg_model = torch.load(bpg_model_dir + "/model", map_location=torch.device(get_device()))

        # Training curves
        training_curves = model_dir + "/training_curves"
        with open(training_curves, "w") as file_obj:
            file_obj.write("Epoch,Validation Accuracy,Validation Loss\n")

        # Training
        logging.info("Training")
        max_acc = 0
        acc = []

        for epoch in tqdm(range(conf_dict["num_epochs"])):
            with open(training_curves, "a") as file_obj:
                current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
                logging.info("Time: {}, Epoch: {}".format(current_time, epoch+1))

                train(experts, optimizers, le, conf_dict, train_list, scaler)
                valid_metrics = validate(experts, le, conf_dict, valid_list, scaler, bpg_model)
                acc.append(valid_metrics["acc"])

                file_obj.write("{},{},{}\n".
                                format(epoch+1, round(valid_metrics['acc'], 3), round(valid_metrics['loss'], 3)))

                # Track the best model
                if valid_metrics['acc'] > max_acc:
                    max_acc = valid_metrics["acc"]
                    for bpg in le_bpg.classes_:
                        torch.save(experts[bpg], model_dir + "/model_" + str(bpg))

                # Stop early if accuracy does not improve over last 10 epochs
                if epoch >= 10:
                    if acc[-1] - acc[-11] < 0.001:
                        logging.info("Detected maximum validation accuracy. Stopping early.")
                        break


if __name__ == '__main__':
    # User inputs
    conf_file = "conf/phone/LSTM_LSTM_rev_mspec_moa_experts.txt"
    num_models = 4

    # Train and validate model
    train_and_validate(conf_file, num_models)
