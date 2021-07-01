from sklearn import preprocessing


def get_label_list(label_type):
    """ Get list of phonetic labels.

    Args:
        label_type (str): type of label to encode

    Returns:
        label_list (list): list of phonetic labels
    """
    # Select file containing label list
    if label_type == "phone":
        label_file = "phones/phone_list.txt"
    elif label_type == "phoneme":
        label_file = "phones/phoneme_list.txt"
    elif label_type == "moa":
        label_file = "phones/moa_list.txt"
    elif label_type == "bpg":
        label_file = "phones/bpg_list.txt"
    elif label_type == "vuv":
        label_file = "phones/vuv_list.txt"
    elif label_type == "moa_vuv":
        label_file = "phones/moavuv_list.txt"

    with open(label_file, 'r') as f:
        label_list = f.readlines()
        for i in range(len(label_list)):
            label_list[i] = label_list[i].replace("\n", "")

    return label_list


def get_label_encoder(label_type):
    """ Get label encoder

    Args:
        label_type (str): type of label to encode (phone or phoneme)

    Returns:
        le (preprocessing.LabelEncoder): label encoder

    """
    # Get list of labels
    label_list = get_label_list(label_type)

    # Get label encoder
    le = preprocessing.LabelEncoder()
    le.fit(label_list)

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
