import numpy as np

# Features
from feature_extraction import fit_normalizer

# Labels
from phone_mapping import get_label_encoder

# Training and testing data
from validation import read_feat_list

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


def tune_hyperparameters(conf_file):
    """ Train and evaluate a phoneme classification model

    Args:
        conf_file (str): txt file containing model info

    """
    # Read in conf file
    conf_dict = read_conf(conf_file)
    #conf_dict["num_hidden"] = 90

    # Label encoder
    le = get_label_encoder(conf_dict["label_type"])

    # Random search over hyperparameters
    hyperparams = {}
    num_combos = 1
    # hyperparams["num_feature_maps"] = [20, 20, 20, 30, 20, 30, 40, 20, 30, 40, 50]
    # hyperparams["max_pooling"] = [(4, 1), (5, 1), (6, 1), (6, 1), (7, 1), (7, 1), (7, 1), (8, 1), (8, 1), (8, 1), (8, 1)]
    # hyperparams["num_hidden"] = [67, 85, 98, 67, 116, 80, 61, 143, 98, 75, 61]
    hyperparams["acc"] = np.zeros((num_combos,))

    for i in range(num_combos):
        print(i)

        # # Set current values of hyperparameters
        # conf_dict["num_feature_maps"] = hyperparams["num_feature_maps"][i]
        # conf_dict["max_pooling"] = hyperparams["max_pooling"][i]
        # conf_dict["num_hidden"] = hyperparams["num_hidden"][i]

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
        logging.info("Training")
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
        hyperparams["acc"][i] = max_acc

    # # Set of hyperparameters that gives the highest accuracy on the validation set
    # best_idx = np.argmax(hyperparams["acc"])
    # best_hyperparams = {}
    # best_hyperparams["num_feature_maps"] = hyperparams["num_feature_maps"][best_idx]
    # best_hyperparams["max_pooling"] = hyperparams["max_pooling"][best_idx]
    # best_hyperparams["num_hidden"] = hyperparams["num_hidden"][best_idx]
    #
    # print(best_hyperparams)


if __name__ == '__main__':
    # Necessary files
    conf_file = "conf/CNN_anechoic_mspec.txt"

    # Train and validate
    tune_hyperparameters(conf_file)
