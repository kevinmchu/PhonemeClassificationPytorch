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

# Evaluation
from performance_metrics import get_performance_metrics

# Labels
from phone_mapping import get_label_encoder

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

            # Encode labels as integers
            y_batch = le.transform(y_batch).astype('long')

            # Move to GPU
            x_batch = (torch.from_numpy(x_batch)).to(device)
            y_batch = (torch.from_numpy(y_batch)).to(device)

            # Get outputs and predictions
            outputs = model(x_batch)
            y_prob = torch.exp(outputs)
            y_pred = torch.argmax(outputs, dim=1)

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(np.array(y_batch.to('cpu')))
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['y_prob']).append((y_prob.to('cpu')).detach().numpy())

    return summary


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

    # Directory in which to save decoding results
    decode_dir = os.path.join(model_dir, "decode", test_set)
    Path(decode_dir).mkdir(parents=True)

    # Configure log file
    logging.basicConfig(filename=decode_dir+"/log", filemode="w", level=logging.INFO)

    # Read in list of feature files
    test_list = read_feat_list(test_feat_list)

    # File containing StandardScaler computed based on the training data
    scale_file = model_dir + "/scaler.pickle"

    # Get predictions
    summary = predict(model, get_label_encoder(conf_dict["label_type"]), conf_dict, test_list, scale_file)

    # Accuracy
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])

    # Get performance metrics
    get_performance_metrics(summary, conf_dict, decode_dir)


if __name__ == '__main__':
    # Inputs
    conf_file = "conf/moa/CNN_rev_mspec.txt"
    model_idx = 4
    test_set = "test_stairway_1_1_3_90"
    feat_type = "mspec"

    test(conf_file, model_idx, test_set, feat_type)
