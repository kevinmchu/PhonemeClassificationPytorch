# External libraries
import numpy as np
from tqdm import tqdm
import collections

# Internal
from file_loader import read_feat_file
from validation import read_feat_list
from phone_mapping import get_label_encoder
from train import read_conf

def unigram(conf_dict):
    """ Generates unigram language model

    This function calculates log probabilities for a unigram
    language model. Note that this language model is for the
    overall sequence, rather than a frame-wise basis.

    Args:
        conf_dict (dict): dictionary of configuration parameters

    Returns:
        lm_dict (dict): dict with language model probabilities
    """
    
    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])
    
    # Get label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Preallocate language model dictionary
    lm_dict = collections.OrderedDict()
    for seq in le.classes_:
        lm_dict[str(seq)] = 0

    #for i in tqdm(range(len(train_list))):
    for i in tqdm(range(20)):
        # Read single feature file
        _, labels = read_feat_file(train_list[i], conf_dict)

        # Get encoded labels
        encoded = le.transform(labels).astype('long')

        # Phoneme transition indices to get overall, rather than framewise sequence
        phone_trans_idx = np.concatenate((np.array([0]), np.where(np.diff(encoded))[0] + 1))
        overall_seq = labels[phone_trans_idx]

        # For each class, update counts
        for y in set(overall_seq):
            lm_dict[y] += np.sum(overall_seq == y)

    # Convert counts into log probs
    num_seq = np.sum(list(lm_dict.values()))
    for key in lm_dict.keys():
        lm_dict[key] = np.log(lm_dict[key]/num_seq)

    return lm_dict


def bigram(conf_dict):
    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])
    
    # Get label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Preallocate language model dictionary
    lm_dict = collections.OrderedDict()
    for i in le.classes_:
        for j in le.classes_:
            # w(k) | w(k-1)
            seq = str(i) + "|" + str(j)
            lm_dict[str(seq)] = 0

    for i in tqdm(range(len(train_list))):
        # Read in single feature file
        _, labels = read_feat_file(train_list[i], conf_dict)

        # Get encoded labels
        encoded = le.transform(labels).astype('long')

        # Phoneme transition indices to get overlal, rather than framewise sequence
        phone_trans_idx = np.concatenate((np.array([0]), np.where(np.diff(encoded))[0] + 1))
        overall_seq = labels[phone_trans_idx]

        # Extract all bi-gram sequences
        bigram_seq = list(map(lambda wk, wk1: wk + "|" + wk1, overall_seq[1:], overall_seq[0:-1]))

        # For each sequence, update counts
        for y in set(bigram_seq):
            lm_dict[y] += np.sum(np.array(bigram_seq) == y)

        # Convert counts into log probs
        num_seq = np.sum(list(lm_dict.values()))
        for key in lm_dict.keys():
            lm_dict[key] = np.log(lm_dict[key]/num_seq)

        return lm_dict


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phoneme/LSTM_sim_rev_fftspec_ci.txt"
    lm_type = "unigram"

    # Read in conf file
    conf_dict = read_conf(conf_file)

    lm_dict = bigram(conf_dict)
    print(lm_dict)
