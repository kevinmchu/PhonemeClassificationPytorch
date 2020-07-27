# phoneMapping.py
# Author: Kevin Chu
# Last Modified: 07/27/2020

from sklearn import preprocessing


def get_phone_list():
    """ Returns phone list

    Returns:
        phones (list): list of phones organized by manner of articulation

    """
    phones = ["pau", "epi", "h#", "pcl", "bcl", "tcl", "dcl", "kcl", "gcl",
              "p", "b", "t", "d", "k", "g", "dx", "q",
              "ch", "jh",
              "s", "z", "sh", "zh", "f", "v", "th", "dh",
              "m", "em", "n", "en", "nx", "ng", "eng",
              "l", "el", "r", "w", "y", "hh", "hv",
              "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"]

    return phones


def get_phoneme_list():
    """ Returns phoneme list

    Returns:
        phonemes (list): list of phonemes organized by manner of articulation

    """
    # Phoneme definitions according to Lee and Hon (note: zh is really sh)
    phonemes = ["sil",
                "p", "b", "t", "d", "k", "g", "dx", "q",
                "ch", "jh",
                "s", "z", "sh", "f", "v", "th", "dh",
                "m", "n", "ng",
                "l", "r", "w", "y", "hh",
                "aa", "ae", "ah", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"]

    return phonemes


def get_moa_list():
    """ Returns manner of articulation list

    Returns:
        moa (list): list of the manners of articulation plus silence

    """
    moa = ["silence", "stop", "affricate", "fricative", "nasal", "semivowel", "vowel"]

    return moa


def get_label_encoder(label_type):
    """ Get label encoder

    Args:
        label_type (str): type of label to encode (phone or phoneme)

    Returns:
        le (preprocessing.LabelEncoder): label encoder

    """
    if label_type == "phone":
        labels = get_phone_list()
    elif label_type == "phoneme":
        labels = get_phoneme_list()
    elif label_type == "moa":
        labels = get_moa_list()

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    return le


def phone_to_phoneme(phones, num_phonemes):
    """ Converts phones into phonemes

    Args:
        phones (list): list of phones
        num_phonemes (int): number of phonemes to reduce to

    Returns:
        phonemes (list): list of phonemes

    """
    # Phone to phoneme mapping
    file = "phones/phone_map_Lee_Hon.txt"

    # Open
    file_obj = open(file, "r")
    x = file_obj.readlines()
    file_obj.close()

    # Creates a dictionary where keys are phones and values are phonemes
    phone_dict = {}
    for i in range(0, len(x)):
        temp = x[i].split()
        if num_phonemes == 48:
            phone_dict[temp[0]] = temp[1]
        elif num_phonemes == 39:
            phone_dict[temp[0]] = temp[2]

    # Convert phones to phonemes
    phonemes = []
    for phone in phones:
        phonemes.append(phone_dict[phone])

    return phonemes


def phone_to_moa(phones):
    """
    This function converts a list of phones into their manner of articulation
    
    Args:
        phones (list): list of phones to convert
        
    Returns:
        moa (list): list of phones converted to manner of articulation
    """
    
    # Phoneme map file
    file = "phones/phone_to_moa.txt"
    
    # Open
    file_obj = open(file, "r")
    x = file_obj.readlines()
    file_obj.close()
    
    # Creates a dictionary where keys are phonemes and values of moa
    phone_dict = {}
    for i in range(0, len(x)):
        temp = x[i].split()
        phone_dict[temp[0]] = temp[1]

    # Converts phonemes to manner of articulation
    moa = []
    for phone in phones:
        moa.append(phone_dict[phone])
    
    return moa
