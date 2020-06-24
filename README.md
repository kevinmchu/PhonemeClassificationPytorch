# PhonemeClassificationPytorch

## About
This repository contains code develop neural network-based phone classification models. The neural networks are implemented using PyTorch, and the features are currently calculated using MATLAB.

## Overview of Toolkit
The main script to run is main.py. This script trains, validates, and tests neural network models on phone classification tasks. The script trains models on a user-specified number of epochs and performs weight updates after each full pass through the training data. 

## Classification Task
This repository can develop neural networks to classify phones, phonemes, or manner of articulation. Mappings from phones to phonemes and manner of articulation are provided in the phones folder.

## Models
The repository currently supports the following neural network architectures:
* MLP
* RNN
* BRNN
* LSTM
* BLSTM

Note: The vanilla RNN models are unstable and often lead to exploding gradients. Therefore, if one prefers to use recurrent models, it would be preferable to use LSTMs.

## Dataset
This repository uses the TIMIT database, which contains speech material from speakers representing eight dialects of American English. The TIMIT database comes with phone alignments obtained through manual means, which are used for the ground truth when training and testing models.
