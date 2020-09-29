from train import get_device
import pickle
import torch
from tqdm import tqdm
from feature_extraction import read_feat_file
import numpy as np
from train import read_conf
import os
from validation import read_feat_list
from pathlib import Path
import logging
from sklearn import preprocessing

# Evaluation
from performance_metrics import get_performance_metrics

# Labels
from phone_mapping import get_label_encoder
from phone_mapping import phone_to_moa
from phone_mapping import get_phone_list

def predict(experts, le, conf_dict, file_list, scale_file, moa_model, moa_label):
    """ Test phoneme classification model

    Args:
        model (torch.nn.Module): neural network model
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        conf_dict (dict): configuration parameters
        file_list (list): files in the test set
        scaler (StandardScaler): scales features to zero mean unit variance

    Returns:
        summary (dict): dictionary containing file name, true class
        predicted class, and probability of predicted class

    """
    logging.info("Testing model")

    # Track file name, true class, predicted class, and prob of predicted class
    summary = {"file": [], "y_true": [], "y_pred": [], "y_prob": []}

    # Get label encoder for phone/phoneme
    le_phone = get_label_encoder(conf_dict["label_type"])

    # If hierarchical classification, get label encoder for moa
    if conf_dict["hierarchical"]:
        le_moa = get_label_encoder("moa")
        unique_moa = np.reshape(le_moa.transform(le_moa.classes_), (1, len(le_moa.classes_)))

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Get the device
    device = get_device()

    # Evaluation mode
    print("Testing")
    moa_model.eval()
    for moa in experts.keys():
        experts[moa].eval()

    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            logging.info("Testing file {}".format(file_list[i]))

            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file_list[i], conf_dict)

            # Normalize features
            x_batch = scaler.transform(x_batch)

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)

            if moa_label == "known":
                y_moa = le_moa.transform(np.array(phone_to_moa(list(y_batch))))
                y_moa = np.reshape(y_moa, (len(y_moa), 1))
                moa_outputs = torch.from_numpy((unique_moa == y_moa).astype('float32')).to(device)

            elif moa_label == "predicted_soft" or moa_label == "predicted_hard":
                # Get moa model outputs
                moa_outputs = torch.exp(moa_model(x_batch))

            if moa_label == "predicted_hard":
                max_probs = torch.max(moa_outputs, dim=1)
                moa_outputs = (moa_outputs == torch.reshape(max_probs[0], (len(max_probs[0]), 1))).float()

            # Get posterior probabilities from each expert
            y_prob = torch.zeros((len(y_batch), len(le_phone.classes_)))
            y_prob = y_prob.to(device)
            for moa in experts.keys():
                outputs = torch.exp(experts[moa](x_batch))
                y_prob[:, le_phone.transform(le[moa].classes_)] = moa_outputs[:, le_moa.transform([moa])] * outputs

            # Get outputs and predictions
            y_pred = torch.argmax(y_prob, dim=1)

            # Encode labels as integers
            y_batch = le_phone.transform(y_batch).astype('long')

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(y_batch)
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['y_prob']).append((y_prob.to('cpu')).detach().numpy())

    return summary


def test(conf_file, model_idx, test_set, feat_type, moa_label):
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

    # Load moa model
    moa_model_dir = os.path.join(conf_dict["moa_model_dir"], "model" + str(model_idx))
    moa_model = torch.load(moa_model_dir + "/model", map_location=torch.device(get_device()))

    # Load moa experts
    model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""),
                             "model" + str(model_idx))
    experts = {}
    le = {}

    phone_list = get_phone_list()
    phone_list_as_moa = phone_to_moa(phone_list)
    le_moa = get_label_encoder("moa")

    for moa in le_moa.classes_:
        # Load experts
        experts[moa] = torch.load(model_dir + "/model_" + moa, map_location=torch.device(get_device()))

        # Get label encoder
        idx = np.argwhere(np.array(phone_list_as_moa) == moa)
        idx = np.reshape(idx, (len(idx),))
        le[moa] = preprocessing.LabelEncoder()
        le[moa].fit(list(np.array(phone_list)[idx]))

    # Directory in which to save decoding results
    decode_dir = os.path.join(model_dir, "decode", moa_label, test_set)
    Path(decode_dir).mkdir(parents=True)

    # Configure log file
    logging.basicConfig(filename=decode_dir+"/log", filemode="w", level=logging.INFO)

    # Read in list of feature files
    test_list = read_feat_list(test_feat_list)

    # File containing StandardScaler computed based on the training data
    scale_file = model_dir + "/scaler.pickle"

    # Get predictions
    summary = predict(experts, le, conf_dict, test_list, scale_file, moa_model, moa_label)

    # Accuracy
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])

    # Get performance metrics
    get_performance_metrics(summary, conf_dict, decode_dir)


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phone/LSTM_LSTM_rev_mspec_hierarchical.txt"
    model_idx = 1
    test_set = "test_anechoic"
    feat_type = "melSpectrogram"
    moa_label = "predicted_hard"

    test(conf_file, model_idx, test_set, feat_type, moa_label)
