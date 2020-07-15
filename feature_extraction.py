# featureExtraction.py
# Author: Kevin Chu
# Last Modified: 06/11/2020

import numpy as np
from sklearn import preprocessing
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa


def fit_normalizer(file_list, label_type):
    """ Fits feature normalizer using list of files

    Args:
        file_list (list): list of training files
        label_type(str): phone or phoneme

    Returns:
        scaler (StandardScaler): scaler estimated on training data
    """

    # Instantiate
    scaler = preprocessing.StandardScaler()

    for file in file_list:
        # Get features
        X, _ = read_feat_file(file, label_type)

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
    
    # Read in features and labels as a string
    file_obj = open(filename, "r")
    x = file_obj.readlines()
    file_obj.close()
    
    # Extract features and labels
    featsAndLabs = list(map(lambda a: a.split(), x))
    X = np.array(list(map(lambda a: a[0:-1], featsAndLabs)), dtype='float32')
    y = list(map(lambda a: a[-1], featsAndLabs))

    # Deltas and delta-deltas calculated causally
    if conf_dict["deltas"]:
        deltas = np.concatenate((np.zeros((1, np.shape(X)[1]), dtype='float32'), np.diff(X, axis=0)), axis=0)
        X = np.concatenate((X, deltas), axis=1)
        if conf_dict["deltaDeltas"]:
            delta_deltas = np.concatenate((np.zeros((1, np.shape(deltas)[1]), dtype='float32'), np.diff(deltas, axis=0)), axis=0)
            X = np.concatenate((X, delta_deltas), axis=1)


    if conf_dict["label_type"] == 'phoneme':
        # Map phones to phonemes
        y = phone_to_phoneme(y, 39)
        y = np.array(y, dtype='object')

    elif conf_dict["label_type"] == 'moa':
        y = phone_to_moa(y)
        y = np.array(y, dtype='object')
    
    return X, y
