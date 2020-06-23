import numpy as np
import matplotlib.pyplot as plt


def plot_outputs(y_prob, y_true, le):
    """ Plot neural network outputs

    This function plots the probability that the current frame is the correct phone.

    Args:
        y_prob (np.array): 2D matrix that contains probs for each phone across time
        y_true (np.array): 1D matrix of framewise labels expressed as ints
        le (LabelEncoder): label encoder that maps ints to phones

    Returns:
        none

    """
    y_prob_correct = np.zeros((len(y_prob),))

    # For each frame, get prob of correctly identifying phone
    for frame in range(len(y_true)):
        y_prob_correct[frame] = y_prob[frame, y_true[frame]]

    phone_trans_idx = np.concatenate((np.array([0]), np.where(np.diff(y_true))[0] + 1, np.array([len(y_true)-1])))
    text_label_idx = (phone_trans_idx[0:-1] + np.round(np.diff(phone_trans_idx)/3)).astype(int)

    # Plot
    plt.plot(y_prob_correct)
    for i, xc in enumerate(phone_trans_idx):
        plt.axvline(x=xc, color='k', linestyle='--')
        if i < len(phone_trans_idx) - 1:
            plt.text(text_label_idx[i], 1, le.inverse_transform(y_true[text_label_idx])[i])

    plt.axis([0, 100, 0, 1])
    plt.xlabel("Frame")
    plt.ylabel("Probability")
    plt.show()

    return
