from train import get_device
import pickle
import torch
from tqdm import tqdm
from file_loader import read_feat_file
import numpy as np
from train import read_conf
import os
from file_loader import read_feat_list
from pathlib import Path
import logging
from sklearn import preprocessing

# Evaluation
from performance_metrics import get_performance_metrics

# Labels
from phone_mapping import get_label_encoder
from phone_mapping import phone_to_moa
from phone_mapping import phone_to_bpg
from phone_mapping import get_phone_list

def predict(experts, le, conf_dict, file_list, scale_file, bpg_model, bpg_label):
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
        le_bpg = get_label_encoder(conf_dict["bpg"])
        unique_bpg = np.reshape(le_bpg.transform(le_bpg.classes_), (1, len(le_bpg.classes_)))

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Get the device
    device = get_device()

    # Evaluation mode
    print("Testing")
    bpg_model.eval()
    for bpg in experts.keys():
        experts[bpg].eval()

    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            logging.info("Testing file {}".format(file_list[i]))

            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file_list[i], conf_dict)

            # Normalize features
            x_batch = scaler.transform(x_batch)

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)

            if bpg_label == "known":
                if conf_dict["bpg"] == "moa":
                    y_bpg = le_bpg.transform(np.array(phone_to_moa(list(y_batch))))
                elif conf_dict["bpg"] == "bpg":
                    y_bpg = le_bpg.transform(np.array(phone_to_bpg(list(y_batch))))

                y_bpg = np.reshape(y_bpg, (len(y_bpg), 1))
                bpg_outputs = torch.from_numpy((unique_bpg == y_bpg).astype('float32')).to(device)

            elif bpg_label == "predicted_soft" or bpg_label == "predicted_hard":
                # Get bpg model outputs
                bpg_outputs = torch.exp(bpg_model(x_batch))

            if bpg_label == "predicted_hard":
                max_probs = torch.max(bpg_outputs, dim=1)
                bpg_outputs = (bpg_outputs == torch.reshape(max_probs[0], (len(max_probs[0]), 1))).float()

            # Get posterior probabilities from each expert
            y_prob = torch.zeros((len(y_batch), len(le_phone.classes_)))
            y_prob = y_prob.to(device)
            for bpg in experts.keys():
                outputs = torch.exp(experts[bpg](x_batch))
                y_prob[:, le_phone.transform(le[bpg].classes_)] = bpg_outputs[:, le_bpg.transform([bpg])] * outputs

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


def test(conf_file, model_idx, test_set, feat_type, bpg_label):
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
    bpg_model_dir = os.path.join(conf_dict["bpg_model_dir"], "model" + str(model_idx))
    bpg_model = torch.load(bpg_model_dir + "/model", map_location=torch.device(get_device()))

    # Load moa experts
    model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""),
                             "model" + str(model_idx))
    experts = {}
    le = {}

    phone_list = get_phone_list()

    if conf_dict["bpg"] == "moa":
        phone_list_as_bpg = phone_to_moa(phone_list)
    elif conf_dict["bpg"] == "bpg":
        phone_list_as_bpg = phone_to_bpg(phone_list)

    le_bpg = get_label_encoder(conf_dict["bpg"])

    for bpg in le_bpg.classes_:
        # Load experts
        experts[bpg] = torch.load(model_dir + "/model_" + bpg, map_location=torch.device(get_device()))

        # Get label encoder
        idx = np.argwhere(np.array(phone_list_as_bpg) == bpg)
        idx = np.reshape(idx, (len(idx),))
        le[bpg] = preprocessing.LabelEncoder()
        le[bpg].fit(list(np.array(phone_list)[idx]))

    # Directory in which to save decoding results
    decode_dir = os.path.join(model_dir, "decode", bpg_label, test_set)

    try:
        Path(decode_dir).mkdir(parents=True)
    except FileExistsError:
        print("File exists: " + decode_dir + ". Overwriting existing files")

    # Configure log file
    logging.basicConfig(filename=decode_dir+"/log", filemode="w", level=logging.INFO)

    # Read in list of feature files
    test_list = read_feat_list(test_feat_list)

    # File containing StandardScaler computed based on the training data
    scale_file = model_dir + "/scaler.pickle"

    # Get predictions
    summary = predict(experts, le, conf_dict, test_list, scale_file, bpg_model, bpg_label)

    # Accuracy
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])

    # Get performance metrics
    get_performance_metrics(summary, conf_dict, decode_dir)


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/phone/LSTM_LSTM_rev_mspec_moa_experts.txt"
    model_idx = 3
    test_set = "test_stairway_1_1_3_90"
    feat_type = "melSpectrogram"
    bpg_label = "predicted_soft"

    test(conf_file, model_idx, test_set, feat_type, bpg_label)
