import numpy as np

# Features
from file_loader import fit_normalizer

# Labels
from phone_mapping import get_label_encoder

# Training and testing data
from file_loader import read_feat_list

# PyTorch
import torch.optim as optim

# Models
from net import initialize_network

from tqdm import tqdm
import logging

from train import get_device
from train import train
from train import validate
from train import read_conf
from train import convert_string


def read_hyperparams(hyperparams_file):
    """ Read in hyperparams file as a dictionary

    Args:
        hyperparams_file (str): name of hyperparameter file

    Returns:
        hyperparams_dict (dict): hyperparameter file expressed as dict

    """

    with open(hyperparams_file, "r") as f:
        x = f.readlines()

    x = list(map(lambda a: a.replace('\n', ''), x))
    x = list(map(lambda a: a.split('\t'), x))

    # Convert list of hyperparameters to dict
    hyperparams_dict = {}
    for key in x[0]:
        hyperparams_dict[key] = []

    for i in range(1, len(x)):
        for j in range(len(x[0])):
            hyperparams_dict[x[0][j]].append(convert_string(x[0][j], x[i][j]))

    return hyperparams_dict


def tune_hyperparameters(conf_file, hyperparams_file):
    """ Train and evaluate a phoneme classification model

    Args:
        conf_file (str): txt file containing model info

    """
    # Read in conf file
    conf_dict = read_conf(conf_file)

    # Read in hyperparams file
    hyperparams_dict = read_hyperparams(hyperparams_file)

    # Track accuracy of each combo of hyperparameters
    num_combos = len(hyperparams_dict[list(hyperparams_dict.keys())[0]])
    hyperparams_acc = np.zeros((num_combos,))

    # Label encoder
    le = get_label_encoder(conf_dict["label_type"])

    for i in range(num_combos):
        # Configure logging
        print("Hyperparameter set: {}".format(i+1))
        logging.basicConfig(filename=hyperparams_file.replace(".txt", "_best.log"), filemode="w", level=logging.INFO)
        msg = "Hyperparameter set: " + str(i+1) + ", "
        for key in hyperparams_dict.keys():
            msg += (str(key) + " = " + str(hyperparams_dict[key][i]) + ", ")

        logging.info(msg)

        # Replace conf_dict with current values of hyperparameters
        for key in hyperparams_dict.keys():
            conf_dict[key] = hyperparams_dict[key][i]

        # Initialize network
        model = initialize_network(conf_dict)

        # Send network to GPU (if applicable)
        device = get_device()
        model.to(device)

        # Stochastic gradient descent with user-defined learning rate and momentum
        optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])

        # Read in feature files
        train_list = read_feat_list(conf_dict["training"])
        valid_list = read_feat_list(conf_dict["development"])

        # Get standard scaler
        scaler = fit_normalizer(train_list, conf_dict)

        # Training
        max_acc = 0
        acc = []

        for epoch in tqdm(range(conf_dict["num_epochs"])):
            train(model, optimizer, le, conf_dict, train_list, scaler)
            valid_metrics = validate(model, le, conf_dict, valid_list, scaler)
            acc.append(valid_metrics["acc"])
            print("Validation Accuracy: {}".format(round(valid_metrics['acc'], 3)))

            # Track the best model
            if valid_metrics['acc'] > max_acc:
                max_acc = valid_metrics["acc"]

            # Stop early if accuracy does not improve over last 10 epochs
            if epoch >= 10:
                if acc[-1] - acc[-11] < 0.001:
                    break

        # Save the accuracy of the best model for current set of hyperparameters
        hyperparams_acc[i] = max_acc
        logging.info("Accuracy = " + str(round(max_acc, 3)))

    # Set of hyperparameters that gives the highest accuracy on the validation set
    best_idx = np.argmax(hyperparams_acc)

    best_hyperparams_file = hyperparams_file.replace(".txt", "_best.txt")
    with open(best_hyperparams_file, 'w') as f:
        for key in hyperparams_dict.keys():
            f.write(str(key) + " = " + str(hyperparams_dict[key][best_idx]) + "\n")


if __name__ == '__main__':
    # Necessary files
    conf_file = "conf/phone/CNN_rev_mspec.txt"
    hyperparams_file = "hyperparams/CNN_rev_mspec_hyperparams.txt"

    # Train and validate
    tune_hyperparameters(conf_file, hyperparams_file)
