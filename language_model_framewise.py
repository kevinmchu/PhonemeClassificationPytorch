# External libraries
import numpy as np
from tqdm import tqdm
import collections
import os
import os.path
from pathlib import Path
import sys

# Internal
from file_loader import read_feat_file
from file_loader import read_feat_list
from phone_mapping import get_label_encoder
from conf import read_conf


def unigram(conf_dict):
    """ Generates unigram language model

    This function calculates the log probabilities for a unigram
    language model.

    Args:
        conf_dict (dict): dictionary of configuration parameters

    Returns:
        lm (np.array): array with language model log probabilities

    """

    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])

    # Get label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Language model array, where each element is a phone
    lm = np.zeros((len(le.classes_), 1))

    for i in tqdm(range(len(train_list))):
        _, labels = read_feat_file(train_list[i], conf_dict)

        labels = le.transform(labels)
        
        # For each class, update counts
        for y in np.unique(labels):
            lm[y] += np.sum(np.array(labels) == y)

    # Convert counts into probs
    lm = lm/np.sum(lm)

    assert (np.sum(lm) > 0.999 and np.sum(lm) < 1.001)
    
    return np.log(lm), le.classes_


def bigram(conf_dict):
    """ Generates bigram language model

    This function calculates the log probabilities for a bigram
    language model.

    Args:
        conf_dict (dict): dictionary of configuration parameters

    Returns:
        lm (np.array): 2D matrix of bigram language model log probs

    """

    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])

    # Get label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Language model matrix
    # Each row is the nth phone, each column is the (n-1)th phone
    lm = np.zeros((len(le.classes_), len(le.classes_)))

    for i in tqdm(range(len(train_list))):
        _, labels = read_feat_file(train_list[i], conf_dict)

        # Extract all two-phone sequences
        idx = np.concatenate((np.expand_dims(np.arange(1,len(labels)), axis=1),
                              np.expand_dims(np.arange(0,len(labels)-1), axis=1)),
                             axis=1)

        seq = le.transform(labels)[idx]

        # For each sequence, update counts
        for y in np.unique(seq, axis=0):
            lm[y[0], y[1]] += np.sum(seq == y)

    # Convert counts into probs
    lm = lm/np.sum(lm, axis=0)

    return np.log(lm), le.classes_


def write_lm(lm, classes, lm_type, lm_dir, conf_file, conf_dict):
    """ Saves language model to npy file

    Args:
        lm (np.array): language model log probabilities
        lm_type (str): language model order
        lm_dir (str): directory in which to save language model
        conf_file (str): name of configuration file
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        None
    """
    save_dir = os.path.join(lm_dir, conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt",""))

    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True)
    
    lm_file = save_dir + "/" + lm_type + ".npz"

    np.savez(lm_file, probs=lm, phones=classes)


def read_lm(lm_type, lm_dir, conf_file, conf_dict):
    """ Read in language model from npy file

    Args:
        lm_type (str): language model order
        lm_dir (str): directory in which to save language model
        conf_file (str): name of congiruation file
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        lm (np.array): language model as a dictionary

    """
    save_dir = os.path.join(lm_dir, conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt",""))
    
    lm_file = save_dir + "/" + lm_type + ".npz"

    data = np.load(lm_file)
    lm = data["probs"]

    # Replace nan and -inf with small negative numbers
    lm[np.isinf(lm)] = np.log(sys.float_info.min)
    lm[np.isnan(lm)] = np.log(sys.float_info.min)

    return lm


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phone/LSTM_rev_mspec.txt"
    lm_type = "2gram"
    lm_dir = "lm"

    # Read in conf file
    conf_dict = read_conf(conf_file)

    # Get bigram probabilities
    if lm_type == "1gram":
        lm, classes = unigram(conf_dict)
    elif lm_type == "2gram":
        lm, classes = bigram(conf_dict)

    write_lm(lm, classes, lm_type, lm_dir, conf_file, conf_dict)

    lm_read = read_lm(lm_type, lm_dir, conf_file, conf_dict)

    print(np.array_equal(lm, lm_read, equal_nan=True))
