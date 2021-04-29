# featureExtraction.py
# Author: Kevin Chu
# Last Modified: 02/01/2021

import numpy as np
from sklearn import preprocessing
from scipy import signal
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
from phone_mapping import phone_to_bpg


def fit_normalizer(file_list, conf_dict):
    """ Fits feature normalizer using list of files

    Args:
        file_list (list): list of training files
        conf_dict (dict): dictionary with neural network architecture

    Returns:
        scaler (StandardScaler): scaler estimated on training data
    """

    # Instantiate
    scaler = preprocessing.StandardScaler()

    for file in file_list:
        # Get features
        X, _ = read_feat_file(file, conf_dict)

        # Update standard scaler
        scaler.partial_fit(X)

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
