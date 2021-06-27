# PhonemeClassificationPytorch

## About
This repository contains code develop framewie phone classification models, which predict phones for each time step of the input. The neural networks are implemented using PyTorch, and the features are currently calculated using MATLAB.

## Overview of Toolkit
This repository contains code to train, validate, and test neural network models on phone classification tasks. The script trains models on a user-specified number of epochs and performs weight updates after each full pass through the training data. 

## Classification Tasks
This repository can develop neural networks to classify 
* Phones
* Phonemes
* Voiced/unvoiced (VUV) phonemes
* Broad phonetic groups (BPG)
* Manner of articulation (MOA)
* Manner of articulation voiced or unvoiced (MOA-VUV)

## Models
### Architectures
The repository currently supports the following neural network architectures:
* MLP
* RNN
* BRNN
* LSTM
* BLSTM

### Configuration Files
The conf directory contains various configuration files that allow the user to define the model hyperparameters and the datasets used to develop the models. Here is an example of a conf file to train an LSTM.

[Architecture]
model_type = LSTM
bidirectional = False
num_hidden = 123
num_classes = 39

[Training]
batch_size = 1
num_epochs = 250
learning_rate = 1e-5
momentum = 0.9
label_type = phoneme
optimizer = sgd

[Datasets]
training = data/train_librispeech_sim_rev/log_fft.txt
development = data/dev_librispeech_sim_rev/log_fft.txt

[Features]
feature_type = fftspec_ci
num_coeffs = 65
use_energy = False
deltas = False
deltaDeltas = False

The Architecture section contains detail regarding the model type, whether the model is unidirectional or bidirectional (for RNNs), and the hyperparameters of the model.

The Training section contains detail regarding the batch size, maximum number of epochs, learning rate, momentum, classification task, and optimizer.

The Datasets section contains the file paths to .txt files that contain lists of feature files.

The Features section contains metadata about the features used to train and test the models.

### Relevant Modules
* `train.py`: trains a model to perform one of the specified classification tasks
* `train_hierarchical.py`: uses Mixture of Experts approach to perform the classification task
* `train_lm.py`: trains a language model
* `train_joint.py`: jointly trains an acoustic model and a language model
* `evaluate.py`: evaluates a trained model
* `evaluate_hierarchical.py`: evaluates trained Mixture of Experts model
* `net.py`: contains model definitions

Note: The vanilla RNN models are unstable and often lead to exploding gradients. Therefore, if one prefers to use recurrent models, it would be preferable to use LSTMs.

## Feature Extraction
### Dataset
This repository uses the TIMIT database, which contains speech material from speakers representing eight dialects of American English. The TIMIT database comes with phone alignments obtained through manual means, which are used for the ground truth when training and testing models.

### Relevant Functions
* `extractFeaturesTrainingData.m`: extracts features for training and validation datasets
* `extractFeaturesTestingData.m`: extracts features for testing dataset
