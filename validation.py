# validation.py
# Author: Kevin Chu
# Last Modified: 05/11/2020

import random


def read_feat_list(feat_list_file):
    # Read in file list
    file_obj = open(feat_list_file, "r")
    file_list = file_obj.readlines()
    file_obj.close()
    
    # Remove newline characters
    file_list = list(map(lambda x: x.replace("\n",""), file_list))
    
    return file_list


def train_val_split(file_list, nval):
    # Randomly shuffle file list
    random.shuffle(file_list)
    
    # Create training/validation split
    val_list = file_list[0:nval]
    train_list = file_list[nval:]
    
    return val_list, train_list

