# language_model.py
# Author: Kevin Chu
# Last Modified: 6/17/20

import numpy as np
from feature_extraction import read_feat_file
from validation import read_feat_list
from phone_mapping import get_label_encoder
from tqdm import tqdm
import collections


def unigram(train_list, label_type):
    """ Generates unigram language model

    This function calculates the log probabilities for a unigram
    language model.

    Args:
        train_list (list): list of feature files of training data
        label_type (str): phone or phoneme

    Returns:
        lm_dict (dict): dict with language model probabilities

    """

    # Get label encoder
    le = get_label_encoder(label_type)

    # Preallocate language model dictionary
    lm_dict = collections.OrderedDict()
    for seq in le.classes_:
        lm_dict[str(seq)] = 0

    for i in tqdm(range(len(train_list))):
        _, labels = read_feat_file(train_list[i], label_type)

        # For each class, update counts
        for y in set(labels):
            lm_dict[y] += np.sum(np.array(labels) == y)

    # Convert counts into log probs
    num_seq = np.sum(list(lm_dict.values()))
    for key in lm_dict.keys():
        lm_dict[key] = np.log(lm_dict[key]/num_seq)

    return lm_dict


def bigram(train_list, label_type):
    """ Generates bigram language model

    This function calculates the log probabilities for a bigram
    language model.

    Args:
        train_list (list): list of feature files of training data
        label_type (str): phone or phoneme

    Returns:
        lm_dict (dict): dict with language model probabilities

    """

    # Get label encoder
    le = get_label_encoder(label_type)

    # Preallocate language model dictionary
    lm_dict = collections.OrderedDict()
    for i in le.classes_:
        for j in le.classes_:
            # w(k) | w(k-1)
            seq = str(i) + "|" + str(j)
            lm_dict[str(seq)] = 0

    for i in tqdm(range(len(train_list))):
        _, labels = read_feat_file(train_list[i], label_type)

        # Extract all two-phone sequences
        seq = list(map(lambda wk, wk1: wk + "|" + wk1, labels[1:], labels[0:-1]))

        # For each sequence, update counts
        for y in set(seq):
            lm_dict[y] += np.sum(np.array(seq) == y)

    # Convert counts into log probs
    num_seq = np.sum(list(lm_dict.values()))
    for key in lm_dict.keys():
        lm_dict[key] = np.log(lm_dict[key]/num_seq)

    return lm_dict


def write_lm(train_list, lm_type, label_type):
    """ Write language model to txt file

    Args:
        train_list (list): list of feature files of training data
        lm_type (str): language model order
        label_type (str): phone or phoneme

    Returns:
        none

    """

    if lm_type is "unigram":
        lm_file = "lm/1gram.txt"
        lm_dict = unigram(train_list, label_type)
    elif lm_type is "bigram":
        lm_file = "lm/2gram.txt"
        lm_dict = bigram(train_list, label_type)

    # Write log probabilities to file
    with open(lm_file, "w") as f:
        for key in lm_dict.keys():
            f.write("%s\t%.4f\n" % (key, lm_dict[key]))


def read_lm(lm_type):
    """ Read in language model from txt file

    Args:
        lm_type (str): language model order

    Returns:
        lm_dict (dict): language model as a dictionary

    """

    if lm_type is "unigram":
        lm_file = "lm/1gram.txt"
    elif lm_type is "bigram":
        lm_file = "lm/2gram.txt"

    # Preallocate
    lm_dict = collections.OrderedDict()

    # Create language model dictionary
    # Keys are phone sequences, and values are log probabilities
    with open(lm_file, "r") as f:
        for line in f:
            line = line.strip("\n")
            line = line.split("\t")
            lm_dict[line[0]] = float(line[1])

    return lm_dict


if __name__ == '__main__':
    # Inputs
    train_feat_list = "data/train.txt"
    lm_type = "bigram"
    label_type = "phone"

    # Read in feature file list
    train_list = read_feat_list(train_feat_list)

    # Generate lm and output to txt file
    #write_lm(train_list, lm_type, label_type)

    lm_dict = read_lm(lm_type)
    print(lm_dict)

