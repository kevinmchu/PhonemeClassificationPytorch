# test_with_lm.py
# Author: Kevin Chu
# Last Modified: 6/23/2020

from decoder import ViterbiDecoder
from feature_extraction import read_feat_file
from language_model import read_lm
from main import get_device
import numpy as np
import pickle
import torch
import tqdm


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
    bigram_probs = torch.reshape(torch.tensor(list(bigram_dict.values())), (len(unigram_probs), len(unigram_probs))).to(
        device)

    # Viterbi decoder
    prior_probs = torch.reshape(unigram_probs, (1, len(unigram_probs)))
    if lm_type is "unigram":
        trans_mat = (torch.reshape(unigram_probs, (len(unigram_probs), 1))).repeat((1, len(unigram_probs)))  # unigram
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
            y_pred, best_score = viterbi_decoder.decode(x_batch)
            y_prob = torch.exp(best_score)

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(np.array(y_batch.to('cpu')))
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['y_prob']).append((y_prob.to('cpu')).detach().numpy())

    return summary
