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
from net import get_model_type

# Evaluation
from performance_metrics import get_performance_metrics

# Labels
from phone_mapping import get_label_encoder
from phone_mapping import phone_to_moa

def predict(model, le, conf_dict, file_list, scale_file):
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

    # If hierarchical classification, get label encoder for moa
    if 'hierarchical' in conf_dict.keys():
        if conf_dict["hierarchical"]:
            le_moa = get_label_encoder("moa")
            unique_moa = np.reshape(le_moa.transform(le_moa.classes_), (1, len(le_moa.classes_)))

    # Get the device
    device = get_device()

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Evaluation mode
    model.eval()
    print("Testing")

    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            logging.info("Testing file {}".format(file_list[i]))

            # Extract features and labels for current file
            x_batch, y_batch = read_feat_file(file_list[i], conf_dict)

            # Normalize features
            x_batch = scaler.transform(x_batch)

            # If hierarchical, add moa feature
            if "hierarchical" in conf_dict.keys():
                if conf_dict["hierarchical"]:
                    blah = 1

            # Encode labels as integers
            y_batch = le.transform(y_batch).astype('long')

            # Reshape to (num_batch, seq_len, num_feats/num_out)
            x_batch = np.reshape(x_batch, (1, np.shape(x_batch)[0], np.shape(x_batch)[1]))

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)

            # Get outputs and predictions
            outputs = model(x_batch)
            y_prob = torch.exp(outputs)
            y_pred = torch.argmax(outputs, dim=2)
            y_pred = torch.squeeze(y_pred, 0)

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(y_batch)
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['y_prob']).append((y_prob.to('cpu')).detach().numpy())

    return summary


def test(conf_file, model_name, test_set):
    """ Make predictions and calculate performance metrics on
    the testing data.

    Args:
        conf_file (str): txt file containing model info
        model_name (str): name of the model
        test_set (str): specifies testing condition

    Returns:
        none

    """

    # Read configuration file
    conf_dict = read_conf(conf_file)

    # List of feature files for the testing data
    test_feat_list = "data/" + test_set + "/" + conf_dict["feature_type"] + ".txt"

    # Load trained model
    model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[2]).replace(".txt", ""),
                             model_name)

    model = get_model_type(conf_dict)
    model.load_state_dict(torch.load(model_dir + "/model.pt"), strict=False)
    #checkpoint = torch.load(model_dir + "/checkpoint.pt")
    #model.load_state_dict(checkpoint['model'], strict=False)

    # Move to GPU
    device = get_device()
    model.to(device)

    # Directory in which to save decoding results
    decode_dir = os.path.join(model_dir, "decode", test_set)

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

    # Label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Get predictions
    summary = predict(model, le, conf_dict, test_list, scale_file)

    # Save decoded results
    save_decoding(summary, test_set, le, decode_dir)

    # Accuracy
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])

    # Get performance metrics
    get_performance_metrics(summary, conf_dict, decode_dir)


def save_decoding(summary, test_set, le, decode_dir):
    """ Saves decoding results

    Args:
        summary (dict): dictionary with true and predicted labels
        test_set (str): name of testing set
        le (LabelEncoder): encodes labels as integers
        decode_dir (str): directory in which to save decodings

    Returns:
        none
    """
    for i, file in enumerate(summary['file']):
        # File where estimated decodings are saved
        save_file = decode_dir + file.split(test_set)[1]

        # Create directory
        out_dir = ("/").join(save_file.split("/")[0:-1])
        try:
            Path(out_dir).mkdir(parents=True)
        except FileExistsError:
            print("File exists: " + out_dir + ". Overwriting existing files")

        to_write = np.hstack((le.inverse_transform(summary['y_true'][i]).reshape(len(summary['y_true'][i]), 1),
                              le.inverse_transform(summary['y_pred'][i]).reshape(len(summary['y_pred'][i]), 1)))

        print(np.shape(to_write))

        np.savetxt(save_file, to_write, fmt='%s', delimiter=' ')


if __name__ == '__main__':
    # # Inputs
    conf_file = "conf/phoneme/LSTM_sim_rev_fftspec_ci.txt"
    model_name = "librispeech"
    test_set = "test_hint_stairway_0_1_3_90"
    
    test(conf_file, model_name, test_set)

    #conf_dir = "conf/phone"
    #conf_files = ["conf/phone/LSTM_rev_mspec.txt"]
    #model_idxs = [0,1,2,3,4]
    #test_sets = ["test_anechoic", "test_office_0_1_3", "test_office_1_1_3", "test_stairway_0_1_3_90", "test_stairway_1_1_3_90"]

    #for conf_file in conf_files:
    #    for model_idx in model_idxs:
    #        for test_set in test_sets:
    #            test(conf_file, model_idx, test_set)
