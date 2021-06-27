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
    # Phoneme definitions according to Lee and Hon, 1989*
    # Note:
    # Lee and Hon (1989) consider 'zh' to be 'sh'
    # *Instead of removing the glottal stop q, it has been mapped to silence
    phonemes = ["sil",
                "p", "b", "t", "d", "k", "g", "dx",
                "ch", "jh",
                "s", "z", "sh", "f", "v", "th", "dh",
                "m", "n", "ng",
                "l", "r", "w", "y", "hh",
                "aa", "ae", "ah", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"]

    return phonemes


def get_moavuv_list():
    """ Returns voiced and unvoiced manner of articulation list                                     
                                                                                                    
    Returns:                                                                                        
        moa_vuv (list): list of v/uv moas plus silence                                              
    """
    moa_vuv = ["silence", "stop_uv", "stop_v", "affricate_uv", "affricate_v", "fricative_uv", "fricative_v","nasal", "semivowel", "vowel"]

    return moa_vuv


def get_moa_list():
    """ Returns manner of articulation list

    Returns:
        moa (list): list of the manners of articulation plus silence

    """
    moa = ["silence", "stop", "affricate", "fricative", "nasal", "semivowel", "vowel"]

    return moa


def get_bpg_list():
    """ Returns broad phonetic group list

    Returns:
        bpg (list): list of broad phonetic groups plus silence

    """
    bpg = ["silence", "stop", "fricative", "nasal", "vowel"]

    return bpg


def get_vuv_list():
    """ Returns voiced/unvoiced group list

    Returns:
        vuv (list): list of voiced/unvoiced groups plus silence

    """
    vuv = ["voiced", "unvoiced", "silence"]

    return vuv
    

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
    elif label_type == "bpg":
        labels = get_bpg_list()
    elif label_type == "vuv":
        labels = get_vuv_list()
    elif label_type == "moa_vuv":
        labels = get_moavuv_list()

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


def map_phones(phones, label_type):
    """
    This function maps phones to reduced sets (other than phoneme)

    Args:
        phones (list): list of phones to convert

    Returns:
        mapped_phones (list): list of phones mapped to reduced sets
    """
    # Select appropriate mapping file
    # Phone to manner of articulation mapping as defined by TIMIT docs
    if label_type == "moa":
        file = "phones/phone_to_moa_timit.txt"
    elif label_type == "bpg":
        file = "phones/phone_to_bpg_timit.txt"
    elif label_type == "vuv":
        file = "phones/phone_to_vuv.txt"
    elif label_type == "moa_vuv":
        file = "phones/phone_to_moa_vuv_timit2.txt"
    
    # Open
    file_obj = open(file, "r")
    x = file_obj.readlines()
    file_obj.close()
    
    # Creates a dictionary where keys are phonemes and values of mapped phones
    phone_dict = {}
    for i in range(0, len(x)):
        temp = x[i].split()
        phone_dict[temp[0]] = temp[1]

    # Converts phonemes to manner of articulation
    mapped_phones = []
    for phone in phones:
        mapped_phones.append(phone_dict[phone])
    
    return mapped_phones
