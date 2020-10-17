import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from train import read_conf
from train import get_device
from evaluate import predict
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
    text_label_idx = (phone_trans_idx[0:-1] + np.round(np.diff(phone_trans_idx)/2)).astype(int)
    phone_trans_idx2 = np.concatenate((np.array([0]), np.where(np.diff(y_pred))[0] + 1, np.array([len(y_pred)-1])))
    text_label_idx2 = (phone_trans_idx2[0:-1] + np.round(np.diff(phone_trans_idx2)/2)).astype(int)

    upper = len(y_pred) # number of time steps to plot
    accuracy = np.round(100 * np.sum(y_true[0:upper+1] == y_pred[0:upper+1])/(upper+1), 1)

    # Plot
    fig, ax = plt.subplots(figsize=(8,3))
    plt.subplots_adjust(bottom=0.2)
    ax.plot(y_prob_correct, 'b', label='truth')
    ax.set_title("asa (Framewise Accuracy = " + str(accuracy) + "%)", fontsize=14)
    ax.set_ylim([0, 1])
    ax.set_ylabel("Probability", fontsize=14)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for i, xc in enumerate(phone_trans_idx[phone_trans_idx < upper]):
        ax.text(xc, -0.05, "|", color='blue')
        if i < len(phone_trans_idx) - 1:
            ax.text(text_label_idx[i], -0.13, abbreviate_moa(le.inverse_transform(y_true[text_label_idx])[i]), color='blue')

    ax.plot(y_prob_max, 'r--', label='prediction')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for i, xc in enumerate(phone_trans_idx2[phone_trans_idx2 < upper]):
        ax.text(xc, -0.2, "|", color='red')
        if i < len(phone_trans_idx2) - 1:
            ax.text(text_label_idx2[i], -0.28, abbreviate_moa(le.inverse_transform(y_pred[text_label_idx2])[i]), color='red')

    plt.xlim([0, upper])
    ax.legend(loc='lower left')

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
        curr_abbrev = 'st'
    elif label == 'affricate':
        curr_abbrev = 'af'
    elif label == 'fricative':
        curr_abbrev = 'f'
    elif label == 'nasal':
        curr_abbrev = 'n'
    elif label == 'semivowel':
        curr_abbrev = "sv"
    elif label == 'vowel':
        curr_abbrev = 'v'
    else:
        curr_abbrev = label

    return curr_abbrev


def write_probs_to_txt(y_true, y_pred, y_prob_phoneme):

    with open('filename.txt', 'w') as f:
        for i in range(len(y_true)):
            f.write(y_true[i] + '\t' + y_pred[i])
            for j in range(np.size(y_prob_phoneme, axis=1)):
                f.write('\t' + str(y_prob_phoneme[i,j]))
            f.write('\n')

    return


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phone/LSTM_rev_mspec.txt"
    model_idx = 0
    test_set = "test_office_0_1_3"
    feat_type = "mspec"

    summary = test(conf_file, model_idx, test_set, feat_type)

    le_phone = get_label_encoder("phone")
    le_phoneme = get_label_encoder("phoneme")
    phone_list = le_phone.classes_
    phoneme_list = phone_to_phoneme(phone_list, 39)
    phoneme_list = np.array(phoneme_list)

    y_true = summary["y_true"][0]
    y_pred = summary["y_pred"][0]
    y_prob_phone = summary["y_prob"][0]
    y_prob_phoneme = np.zeros((np.shape(y_prob_phone)[0], len(le_phoneme.classes_)))

    for i, phoneme in enumerate(le_phoneme.classes_):
        idx = np.where(phoneme_list == phoneme)
        idx = np.reshape(idx, (np.shape(idx)[0]*np.shape(idx)[1]))
        y_prob_phoneme[:, le_phoneme.transform([phoneme])[0]] = np.sum(y_prob_phone[:, idx], axis=1)

    y_true = phone_to_phoneme(le_phone.inverse_transform(y_true), 39)
    y_pred = phone_to_phoneme(le_phone.inverse_transform(y_pred), 39)

    write_probs_to_txt(y_true, y_pred, y_prob_phoneme)

    #y_true = le_phoneme.transform(y_true)
    #y_pred = le_phoneme.transform(y_pred)

    #plot_outputs(y_prob_phoneme, y_true, y_pred, le_phoneme)
    #plot_outputs(y_prob_phoneme, y_pred, le_phoneme)
