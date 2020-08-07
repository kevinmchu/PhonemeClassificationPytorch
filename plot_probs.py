import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from train import read_conf
from train import get_device
from test import predict
from validation import read_feat_list
from phone_mapping import get_label_encoder
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
from phone_mapping import get_phone_list
from phone_mapping import get_phoneme_list


def plot_outputs(y_prob, y_true, y_pred, le):
    """ Plot neural network outputs

    This function plots the probability that the current frame is the correct phone.

    Args:
        y_prob (np.array): 2D matrix that contains probs for each phone across time
        y_true (np.array): 1D matrix of framewise labels expressed as ints
        y_pred (np.array): 1D matrix of framewise labels expressed as ints
        le (LabelEncoder): label encoder that maps ints to phones

    Returns:
        none

    """
    y_prob_correct = np.zeros((len(y_prob),)) # probability of true label
    y_prob_max = np.zeros((len(y_prob),)) # probability of predicted label

    # For each frame, get prob of correctly identifying phone
    for frame in range(len(y_true)):
        y_prob_correct[frame] = y_prob[frame, y_true[frame]]
        y_prob_max[frame] = y_prob[frame, y_pred[frame]]

    phone_trans_idx = np.concatenate((np.array([0]), np.where(np.diff(y_true))[0] + 1, np.array([len(y_true)-1])))
    text_label_idx = (phone_trans_idx[0:-1] + np.round(np.diff(phone_trans_idx)/3)).astype(int)
    phone_trans_idx2 = np.concatenate((np.array([0]), np.where(np.diff(y_pred))[0] + 1, np.array([len(y_pred)-1])))
    text_label_idx2 = (phone_trans_idx2[0:-1] + np.round(np.diff(phone_trans_idx2)/3)).astype(int)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(y_prob_correct, 'b', label='truth')
    ax1.set_title("Artificial intelligence is ...")
    ax1.set_ylim([0, 1])
    ax1.set_ylabel("Probability")
    ax1.legend()
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for i, xc in enumerate(phone_trans_idx):
        ax1.text(xc, -0.1, "|", color='blue')
        if i < len(phone_trans_idx) - 1:
            ax1.text(text_label_idx[i], -0.2, abbreviate_moa(le.inverse_transform(y_true[text_label_idx])[i]), color='blue')

    ax2.plot(y_prob_max, 'r--', label='prediction')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Probability")
    ax2.legend()
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for i, xc in enumerate(phone_trans_idx2):
        ax2.text(xc, -0.1, "|", color='red')
        if i < len(phone_trans_idx2) - 1:
            ax2.text(text_label_idx2[i], -0.2, abbreviate_moa(le.inverse_transform(y_pred[text_label_idx2])[i]), color='red')

    plt.xlim([0, 138])

    plt.show()

    return


def test(conf_file, model_idx, test_set, feat_type):
    """ Make predictions and calculate performance metrics on
    the testing data.

    Args:
        conf_file (str): txt file containing model info
        model_idx (int): instance of the model
        test_set (str): specifies testing condition
        feat_type (str): mspec or mfcc

    Returns:
        none

    """

    # Read configuration file
    conf_dict = read_conf(conf_file)

    # List of feature files for the testing data
    test_feat_list = "data/" + test_set + "/" + feat_type + ".txt"

    # Load trained model
    model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""),
                             "model" + str(model_idx))
    model = torch.load(model_dir + "/model", map_location=torch.device(get_device()))

    # Read in list of feature files
    test_list = read_feat_list(test_feat_list)

    # File containing StandardScaler computed based on the training data
    scale_file = model_dir + "/scaler.pickle"

    # Get predictions
    summary = predict(model, get_label_encoder(conf_dict["label_type"]), conf_dict, test_list, scale_file)

    return summary


def abbreviate_moa(label):

    if label == 'silence':
        curr_abbrev = 'sil'
    elif label == 'stop':
        curr_abbrev = 'stp'
    elif label == 'affricate':
        curr_abbrev = 'aff'
    elif label == 'fricative':
        curr_abbrev = 'frc'
    elif label == 'nasal':
        curr_abbrev = 'nsl'
    elif label == 'semivowel':
        curr_abbrev = "svw"
    elif label == 'vowel':
        curr_abbrev = 'vwl'

    return curr_abbrev


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phone/LSTM_rev_mspec.txt"
    model_idx = 0
    test_set = "test_office_0_1_3"
    feat_type = "mspec"

    summary = test(conf_file, model_idx, test_set, feat_type)

    le_phone = get_label_encoder("phone")
    le_phoneme = get_label_encoder("moa")
    phone_list = le_phone.classes_
    phoneme_list = phone_to_moa(phone_list)
    phoneme_list = np.array(phoneme_list)

    y_true = summary["y_true"][3]
    y_pred = summary["y_pred"][3]
    y_prob_phone = summary["y_prob"][3]
    y_prob_phoneme = np.zeros((np.shape(y_prob_phone)[0], len(le_phoneme.classes_)))

    for i, phoneme in enumerate(le_phoneme.classes_):
        idx = np.where(phoneme_list == phoneme)
        idx = np.reshape(idx, (np.shape(idx)[0]*np.shape(idx)[1]))
        y_prob_phoneme[:, le_phoneme.transform([phoneme])[0]] = np.sum(y_prob_phone[:, idx], axis=1)

    y_true = phone_to_moa(le_phone.inverse_transform(y_true))
    y_pred = phone_to_moa(le_phone.inverse_transform(y_pred))
    y_true = le_phoneme.transform(y_true)
    y_pred = le_phoneme.transform(y_pred)

    plot_outputs(y_prob_phoneme, y_true, y_pred, le_phoneme)
    #plot_outputs(y_prob_phoneme, y_pred, le_phoneme)
