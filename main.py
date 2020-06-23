import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os.path
import pickle

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
from net import MLP
from net import RNNModel
from net import LSTMModel
from net import initialize_weights

# Language model
from decoder import ViterbiDecoder
from language_model import read_lm
from language_model import write_lm

# Evaluation
from sklearn.metrics import confusion_matrix
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
from confusion_matrix import sort_classes
from confusion_matrix import plot_confusion_matrix
from plot_probs import plot_outputs

from tqdm import tqdm
import logging

def get_device():
    """

    Returns:

    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


def train(model, optimizer, le, label_type, file_list):
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
    scale_file = "features/scaler.pickle"
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


def validate(model, le, label_type, file_list):
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
    scale_file = "features/scaler.pickle"
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


def test(model, lm_type, le, label_type, file_list):
    """ Test phoneme classification model
    
    Args:
        model (torch.nn.Module): neural network model
        lm_type (str): language model type
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        label_type (str): label type
        file_list (list): files in the test set
        
    Returns:
        summary (dict): dictionary containing file name, true class
        predicted class, and probability of predicted class
    
    """

    # Track file name, true class, predicted class, and prob of predicted class
    summary = {"file": [], "y_true": [], "y_pred": [], "y_prob": []}

    # Get the device
    device = get_device()

    # Language model
    unigram_dict = read_lm("unigram")
    unigram_probs = torch.tensor(list(unigram_dict.values())).to(device)
    bigram_dict = read_lm("bigram")
    bigram_probs = torch.reshape(torch.tensor(list(bigram_dict.values())), (len(unigram_probs), len(unigram_probs))).to(device)

    # Viterbi decoder
    prior_probs = torch.reshape(unigram_probs, (1, len(unigram_probs)))
    if lm_type is "unigram":
        trans_mat = (torch.reshape(unigram_probs, (len(unigram_probs), 1))).repeat((1, len(unigram_probs))) # unigram
    elif lm_type is "bigram":
        trans_mat = bigram_probs
    viterbi_decoder = ViterbiDecoder(prior_probs, trans_mat, model)

    # Get scaler
    scale_file = "features/scaler.pickle"
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Evaluation mode
    model.eval()
    print("Testing")

    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file_list[i], label_type)

            # Normalize features
            x_batch = scaler.transform(x_batch)

            # Encode labels as integers
            y_batch = le.transform(y_batch).astype('long')

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)
            y_batch = (torch.from_numpy(y_batch)).to(device)

            # Get outputs and predictions
            #outputs = model(x_batch)
            #y_prob = torch.exp(outputs)
            #y_pred = torch.argmax(outputs, dim=1)
            y_pred, best_score = viterbi_decoder.decode(x_batch)
            y_prob = torch.exp(best_score)

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(np.array(y_batch.to('cpu')))
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['y_prob']).append((y_prob.to('cpu')).detach().numpy())

    return summary


def train_and_validate(model_type, train_list, valid_list, label_type):
    """ Train and evaluate a phoneme classification model

    Args:
        model_type (str): model type
        train_list (list): list of training files
        valid_list (list): list of validation files
        label_type (str): phone or phoneme

    """
    # Get standard scaler
    scale_file = "features/scaler.pickle"
    if not os.path.exists(scale_file):
        scaler = fit_normalizer(train_list, label_type)
        with open(scale_file, 'wb') as f:
            pickle.dump(scaler, f)

    # Label encoder
    le = get_label_encoder(label_type)

    for i in range(3):

        # Instantiate the network
        if model_type is "MLP":
            model = MLP(26, 250, len(le.classes_))
            num_epochs = 250
        elif model_type is "RNN":
            model = RNNModel(26, 275, len(le.classes_), False)
            num_epochs = 120
        elif model_type is "BRNN":
            model = RNNModel(26, 185, len(le.classes_), True)
        elif model_type is "LSTM":
            model = LSTMModel(26, 140, len(le.classes_), False)
            num_epochs = 15
        elif model_type is "BLSTM":
            model = LSTMModel(26, 93, len(le.classes_), True)
            num_epochs = 20

        # Initialize weights
        model.apply(initialize_weights)

        # Send network to GPU (if applicable)
        device = get_device()
        model.to(device)

        # Training parameters
        learn_rate = 1e-5
        m = 0.9

        # Stochastic gradient descent with user-defined learning rate and momentum
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=m)

        # Configure log file
        logging.basicConfig(filename="log/" + model_type + str(i), filemode="w", level=logging.INFO)
        logging.info("Epoch,Training Accuracy,Training Loss,Validation Accuracy,Validation Loss")

        # Training
        print("Training")
        for epoch in tqdm(range(num_epochs)):
            train(model, optimizer, le, label_type, train_list)
            train_metrics = validate(model, le, label_type, train_list)
            valid_metrics = validate(model, le, label_type, valid_list)

            logging.info("{},{},{},{},{}".
                         format(epoch+1, round(train_metrics['acc'], 3), round(train_metrics['loss'], 3),
                                round(valid_metrics['acc'], 3), round(valid_metrics['loss'], 3)))

        # Save model
        torch.save(model, "models/" + model_type + str(i))


if __name__ == '__main__':
    # Necessary files
    train_feat_list = "data/train_gs.txt"
    test_feat_list = "data/test.txt"

    # Parameters
    model_type = "LSTM"
    model_idx = 0
    lm_type = "unigram"
    label_type = "phone"
    num_valid_utts = 184

    # Read in feature list files
    train_list = read_feat_list(train_feat_list)
    test_list = read_feat_list(test_feat_list)

    # Language model
    if not os.path.exists("lm/1gram.txt"):
        write_lm(train_list, "unigram", label_type)

    # Split list of utterances into training and validation sets
    #valid_list, train_list = train_val_split(train_list, num_valid_utts)

    # Train and validate
    #train_and_validate(model_type, train_list, test_list, label_type)

    # # Testing
    # model_name = "models/" + model_type + str(model_idx)
    # model = torch.load(model_name, map_location=torch.device('cpu'))
    # summary = test(model, lm_type, get_label_encoder(label_type), label_type, test_list)
    #
    # # Save
    # save_file = "results/" + model_type + "/" + model_type + str(model_idx) + "_" + lm_type
    # with open(save_file, 'wb') as f:
    #     pickle.dump(summary, f)

    # Load
    save_file = "results/" + model_type + "/" + model_type + str(model_idx) + "_" + lm_type
    with open(save_file, 'rb') as f:
        summary = pickle.load(f)

    y_prob = summary['y_prob'][0]
    y_true = summary['y_true'][0]
    #plot_outputs(y_prob, y_true, get_label_encoder(label_type))
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])

    # # Accuracy
    # accuracy = float(np.sum(summary['y_true'] == summary['y_pred'])) / len(summary['y_true'])
    # print("Accuracy: ", round(accuracy, 3))

    # Plot phone confusion matrix
    le_phone = get_label_encoder(label_type)
    # plot_confusion_matrix(summary['y_true'], summary['y_pred'], le_phone, get_phone_list())

    # Phoneme confusion matrix
    phoneme_true = phone_to_phoneme(le_phone.inverse_transform(summary['y_true']), 39)
    phoneme_pred = phone_to_phoneme(le_phone.inverse_transform(summary['y_pred']), 39)
    le_phoneme = preprocessing.LabelEncoder()
    phoneme_true = le_phoneme.fit_transform(phoneme_true)
    phoneme_pred = le_phoneme.transform(phoneme_pred)
    plot_confusion_matrix(phoneme_true, phoneme_pred, le_phoneme, get_phoneme_list())

    # # Manner of articulation confusion matrixa
    # moa_true = phone_to_moa(le_phone.inverse_transform(summary['y_true']))
    # moa_pred = phone_to_moa(le_phone.inverse_transform(summary['y_pred']))
    # le_moa = preprocessing.LabelEncoder()
    # moa_true = le_moa.fit_transform(moa_true)
    # moa_pred = le_moa.transform(moa_pred)
    # plot_confusion_matrix(moa_true, moa_pred, le_moa, get_moa_list())
