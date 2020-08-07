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

    # Plot
    plt.plot(y_prob_correct, 'b')
    plt.plot(y_prob_max, 'r--')
    for i, xc in enumerate(phone_trans_idx):
        plt.text(xc, 1, "|", color='blue')
        #plt.axvline(x=xc, color='k', linestyle='--')
        if i < len(phone_trans_idx) - 1:
            plt.text(text_label_idx[i], 1, le.inverse_transform(y_true[text_label_idx])[i], color='blue')

    plt.xlim([0, len(y_prob)])
    plt.ylim([0, 1])
    plt.xlabel("Frame")
    plt.ylabel("Probability")
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


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phone/LSTM_rev_mspec.txt"
    model_idx = 0
    test_set = "test_anechoic"
    feat_type = "mspec"

    summary = test(conf_file, model_idx, test_set, feat_type)

    le_phone = get_label_encoder("phone")
    le_phoneme = get_label_encoder("phoneme")
    phone_list = le_phone.classes_
    phoneme_list = phone_to_phoneme(phone_list, 39)
    phoneme_list = np.array(phoneme_list)

    y_true = summary["y_true"][3]
    y_pred = summary["y_pred"][3]
    y_prob_phone = summary["y_prob"][3]
    y_prob_phoneme = np.zeros((np.shape(y_prob_phone)[0], len(le_phoneme.classes_)))

    for i, phoneme in enumerate(le_phoneme.classes_):
        idx = np.where(phoneme_list == phoneme)
        idx = np.reshape(idx, (np.shape(idx)[0]*np.shape(idx)[1]))
        y_prob_phoneme[:, le_phoneme.transform([phoneme])[0]] = np.sum(y_prob_phone[:, idx], axis=1)

    y_true = phone_to_phoneme(le_phone.inverse_transform(y_true), 39)
    y_pred = phone_to_phoneme(le_phone.inverse_transform(y_pred), 39)
    y_true = le_phoneme.transform(y_true)
    y_pred = le_phoneme.transform(y_pred)

    plot_outputs(y_prob_phoneme, y_true, y_pred, le_phoneme)
    #plot_outputs(y_prob_phoneme, y_pred, le_phoneme)
