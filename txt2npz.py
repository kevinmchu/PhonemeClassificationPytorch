import numpy as np
from file_loader import read_feat_list
from tqdm import tqdm
import os


def convert_featfile_to_npz(dataset):
    """ This function converts feature files from .txt format
    to .npz format. This allows the features to be loaded around
    three times faster.

    Args:
        dataset (str): .txt file containing list of features files
        to convert

    Returns:
        None
    """
    feat_list = read_feat_list(dataset)
    
    for i in tqdm(range(len(feat_list))):
        # Open
        file_obj = open(feat_list[i], "r")
        x = file_obj.readlines()
        file_obj.close()

        # Extract features and labels
        featsAndLabs = list(map(lambda a: a.split(), x))
        X = np.array(list(map(lambda a: a[0:65], featsAndLabs)), dtype='float32')
        y = np.array(list(map(lambda a: a[65:-1], featsAndLabs)), dtype='float32')
        phones = np.array(list(map(lambda a: a[-1], featsAndLabs)))

        npz_file = feat_list[i].replace(".txt", ".npz")

        np.savez_compressed(npz_file, feats=X, mask=y, phones=phones)


if __name__ == '__main__':
    # User inputs
    dataset = "data/test_hint_office_0_1_3/fftspec_ci.txt"

    convert_featfile_to_npz(dataset)
