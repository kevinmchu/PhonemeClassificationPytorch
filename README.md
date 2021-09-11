# PhonemeClassificationPytorch

## About
This repository contains code develop framewise phone classification models, which predict phones for each time step of the input. The neural networks are implemented using PyTorch, and the features are currently calculated using MATLAB.

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

## Steps to Training and Testing Classification Models
### Training
1) In the feature_extraction directory, run `extractFeaturesTrainingData.m`. This is a MATLAB script that extracts features for the training and development sets. Refer to comments in the .m file for information on how to define the speech corpus and the RIR database to use for generating the training and development sets. The extracted features are stored in .txt files. IMPORTANT: Make sure to store the feature files in a server with lots of space. The total size of the feature files can easily be in the tens of GB.
2) Run `txt2npy.py`. This is a Python module that converts the feature files from .txt files to compressed .npz files, which allows the feature files to be read in much faster than .txt files, which subsequently allows the models to train much faster.
3) Create a model configuration file using either an existing model in the conf directory or by defining your own model. Additional detail is provided below in the section titled 'Configuration Files'.
4) Run the `train.py` script to train a classification model.

### Testing
The steps to test trained classification models are similar.
1) In the feature_extraction directory, run `extractFeaturesTestingData.m` to extract features for the testing set.
2) Run `txt2npy.py`.
3) Run `evaluate.py` to test the classification models.

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

[Architecture]<br/>
model_type = LSTM<br/>
bidirectional = False<br/>
num_hidden = 123<br/>
num_classes = 39<br/>

[Training]<br/>
batch_size = 1<br/>
num_epochs = 250<br/>
learning_rate = 1e-5<br/>
momentum = 0.9<br/>
label_type = phoneme<br/>
optimizer = sgd<br/>

[Datasets]<br/>
training = data/train_librispeech_sim_rev/log_fft.txt<br/>
development = data/dev_librispeech_sim_rev/log_fft.txt<br/>

[Features]<br/>
feature_type = fftspec_ci<br/>
num_coeffs = 65<br/>
use_energy = False<br/>
deltas = False<br/>
deltaDeltas = False<br/>

The **Architecture** section contains detail regarding the model type, whether the model is unidirectional or bidirectional (for RNNs), and the hyperparameters of the model.

The **Training** section contains detail regarding the batch size, maximum number of epochs, learning rate, momentum, classification task, and optimizer.

The **Datasets** section contains the file paths to .txt files that contain lists of feature files.

The **Features** section contains metadata about the features used to train and test the models.

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
