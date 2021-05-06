# featureExtraction.py
# Author: Kevin Chu
# Last Modified: 02/01/2021

import numpy as np
from sklearn import preprocessing
from scipy import signal
import torch

from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
from phone_mapping import phone_to_bpg


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, conf_dict, le=None):
        self.file_list = file_list
        self.conf_dict = conf_dict
        self.le = le

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Read single feature file
        X, y = read_feat_file(self.file_list[index], self.conf_dict)

        # Map phones to categories and encode as integers
        if bool(self.le):
            y = self.le.transform(y).astype('long')

        return X, y


def collate_fn(batch):
    """
    Used to collate multiple files for dataloader
    """
    X_batch, y_batch = zip(*batch)
    X_batch = list(X_batch)
    y_batch = list(y_batch)

    # Get maximum sequence length
    seq_lens = np.array(list(map(lambda a: len(a), y_batch)), dtype='int')
    max_seq_len = np.max(seq_lens)

    # Pad features with large number and labels with zero
    for file_idx in range(len(y_batch)):
        if np.shape(y_batch[file_idx])[0] < max_seq_len:
            pad_len = max_seq_len - np.shape(y_batch[file_idx])[0]
            X_batch[file_idx] = np.concatenate((X_batch[file_idx], 1000*np.ones((pad_len, np.shape(X_batch[file_idx])[1]), dtype='float32')), axis=0)
            y_batch[file_idx] = np.concatenate((y_batch[file_idx], np.zeros((pad_len,), dtype='long')), axis=0)

    # Convert to np.array
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)

    # T-F bins that are not padded, used to calculate loss
    non_padded = np.prod(X_batch != 1000, axis=2)

    return X_batch, y_batch, non_padded


class TorchStandardScaler:
    """
    Standard scaler for PyTorch tensors
    """

    def __init__(self, mean, var, device):
        self.mean = (torch.tensor(mean.astype('float32')).unsqueeze(0)).to(device)
        self.var = (torch.tensor(var.astype('float32')).unsqueeze(0)).to(device)

    def transform(self, x):
        """ Scales features to zero mean and unit variance

        Args:
            x (torch.Tensor): (batch x seq_len x nfeats)

        Returns:
            x_norm (torch.Tensor): scaled features
        """
        x_norm = (x - self.mean)/torch.sqrt(self.var)

        return x_norm


def fit_normalizer(file_list, conf_dict):
    """ Fits feature normalizer using list of files

    Args:
        file_list (list): list of training files
        conf_dict (dict): dictionary with neural network architecture

    Returns:
        scaler (StandardScaler): scaler estimated on training data
    """

    # Instantiate scaler
    scaler = preprocessing.StandardScaler()

    # Generator for loading utterances
    data_set = Dataset(file_list, conf_dict)
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=conf_dict["batch_size"],
                                                 num_workers=4, collate_fn=collate_fn, shuffle=True)

    # Read in one batch at a time and update scaler
    for X_batch, _, _ in data_generator:
        # Reshape
        X_batch = np.reshape(X_batch, (np.shape(X_batch)[0]*np.shape(X_batch)[1], np.shape(X_batch)[2]))

        # Update scaler using valid indices (i.e. not inf)
        scaler.partial_fit(X_batch[np.where(np.prod(X_batch != np.inf, axis=1))])

    return scaler
    

def read_feat_file(filename, conf_dict):
    """ 
    This function reads features and labels from a text file.
    
    Args:
        filename (str): name of text file containing features and labels
        label_type (str): label type
        
    Returns:
        X (np.array): matrix of features
        y (list): list of phoneme labels
    """
    
    # If disconnected, keep trying to read from file
    io_flag = True
    while io_flag:
        try:
            # Read in features and labels as a string
            file_obj = open(filename, "r")
            x = file_obj.readlines()
            file_obj.close()
            io_flag = False
        except:
            continue
    
    # Extract features and labels
    featsAndLabs = list(map(lambda a: a.split(), x))
    X = np.array(list(map(lambda a: a[0:conf_dict["num_coeffs"]], featsAndLabs)), dtype='float32')
    y = list(map(lambda a: a[-1], featsAndLabs))

    # Train on speech enhanced using irm, if applicable
    if "train_condition" in conf_dict.keys():
        if conf_dict["train_condition"] == "irm":
            irm = np.array(list(map(lambda a: a[conf_dict["num_coeffs"]:-1], featsAndLabs)), dtype='float32')
            X *= irm
    
    # Deltas and delta-deltas
    if conf_dict["deltas"]:
        deltas = calculate_deltas(conf_dict, X)
        X = np.concatenate((X, deltas), axis=1)
        if conf_dict["deltaDeltas"]:
            delta_deltas = calculate_deltas(conf_dict, deltas)
            X = np.concatenate((X, deltaDeltas), axis=1)

    # Splice
    if "window_size" in conf_dict.keys():
        X = splice(X, conf_dict)

    if conf_dict["label_type"] == 'phoneme':
        # Map phones to phonemes
        y = phone_to_phoneme(y, 39)
        y = np.array(y, dtype='object')

    elif conf_dict["label_type"] == 'moa':
        y = phone_to_moa(y)
        y = np.array(y, dtype='object')

    elif conf_dict["label_type"] == 'bpg':
        y = phone_to_bpg(y)
        y = np.array(y, dtype='object')
    
    return X, y


def calculate_deltas(conf_dict, X):
    """
    Calculate deltas from feature matrix. Calculates delta-deltas if given
    deltas as the input.

    Args:
        conf_dict (dict): dictionary with configuration parameters
        X (np.array): feature matrix (can be raw features or deltas)

    Returns:
        deltas (np.array): deltas
    """

    # Number of features in feature matrix
    n_feats = np.shape(X)[1]

    # If causality not specified, assume causal
    if "causal_deltas" not in conf_dict.keys():
        conf_dict["causal_deltas"] = True

    # Causal
    if conf_dict["causal_deltas"]:
        # Current frame minus previous frame
        kernel = np.array([[1.],[-1.]])

        # Kernel index that corresponds to the current frame
        idx_curr_frame = 1

        # Calculate deltas for regions where kernel completely overlaps with feature array
        deltas = signal.convolve2d(kernel, X, mode='valid')

    # Non-causal
    else:
        # Calculate deltas over +/- 2 frames
        kernel = np.array([[2.],[1.],[0.],[-1.],[-2.]])

        # Kernel index that corresponds to the current frame
        idx_curr_frame = 1

        # Calculate deltas for regions where kernel completely overlaps with feature array
        deltas = signal.convolve2d(kernel, X, mode='valid')/np.sum(kernel**2)

    # Apply zero padding for regions where kernel does not completely overlap with feature array
    deltas = np.concatenate((np.zeros((idx_curr_frame, n_feats)), deltas, np.zeros((len(kernel)-1-idx_curr_frame, n_feats))))

    # Cast to float32
    deltas = np.float32(deltas)

    return deltas


def splice(X, conf_dict):
    """
    This function concatenates a feature matrix with features
    from causal time frames. The purpose of this is to increase
    the amount of temporal context available to the model.

    Args:
        X (np.array): matrix of features from current time frame
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        X (np.array): feature matrix with causal time frames
        concatenated
    """

    if "window_size" in conf_dict.keys():
        x0 = np.zeros((conf_dict["window_size"]-1, np.shape(X)[1]), dtype='float32')
        X = np.concatenate((x0, X), axis=0)

        # Splice                                                                                                  
        batch_sz = np.shape(X)[0] - conf_dict["window_size"] + 1
        idx = np.linspace(0, conf_dict["window_size"]-1, conf_dict["window_size"])
        idx = np.tile(idx, (batch_sz, 1)) + np.linspace(0, batch_sz-1, batch_sz).reshape((batch_sz, 1))
        idx = idx.astype(int)
        X = X[idx, :]
        X = X.reshape(np.shape(X)[0], conf_dict["num_features"])

    return X
