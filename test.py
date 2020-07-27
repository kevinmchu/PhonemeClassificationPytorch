from train import get_device
import pickle
import torch
from tqdm import tqdm
from feature_extraction import read_feat_file
import numpy as np
from train import read_conf
import os
from validation import read_feat_list

# Evaluation
from confusion_matrix import plot_confusion_matrix
from confusion_matrix import plot_phoneme_confusion_matrix
from confusion_matrix import plot_moa_confusion_matrix

# Labels
from phone_mapping import get_phone_list
from phone_mapping import get_phoneme_list
from phone_mapping import get_moa_list
from phone_mapping import get_label_encoder

def test(model, le, conf_dict, file_list, scale_file):
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


if __name__ == '__main__':
    # Necessary files
    conf_file = "conf/LSTM_rev_mspec.txt"
    model_idx = 2
    test_feat_list = "data/train_anechoic/mspec.txt"

    # Read configuration file
    conf_dict = read_conf(conf_file)

    # Testing
    model_dir = os.path.join("exp", conf_dict["label_type"], (conf_file.split("/")[1]).replace(".txt", ""),
                             "model" + str(model_idx))
    model = torch.load(model_dir + "/model", map_location=torch.device(get_device()))

    test_list = read_feat_list(test_feat_list)
    scale_file = model_dir + "/scaler.pickle"

    summary = test(model, get_label_encoder(conf_dict["label_type"]), conf_dict, test_list, scale_file)
    y_prob = summary['y_prob'][0]
    y_true = summary['y_true'][0]
    #plot_outputs(y_prob, y_true, get_label_encoder(label_type))
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])

    # Accuracy
    accuracy = float(np.sum(summary['y_true'] == summary['y_pred'])) / len(summary['y_true'])
    print("Accuracy: ", round(accuracy, 3))

    # Plot phone confusion matrix
    le_phone = get_label_encoder(conf_dict["label_type"])
    #plot_confusion_matrix(summary['y_true'], summary['y_pred'], le_phone, get_phone_list())
    plot_phoneme_confusion_matrix(summary['y_true'], summary['y_pred'], le_phone)
    #plot_moa_confusion_matrix(summary['y_true'], summary['y_pred'], le_phone)
