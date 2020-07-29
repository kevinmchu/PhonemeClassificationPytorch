# confusionMatrix.py
# Author: Kevin Chu
# Last Modified: 07/28/2020

import matplotlib.pyplot as plt
import numpy as np
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
from phone_mapping import get_label_encoder
from phone_mapping import get_phone_list
from phone_mapping import get_phoneme_list
from phone_mapping import get_moa_list
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


def sort_classes(cm, classes, sorted_classes):
	""" Sort classes so confusion matrix is in certain order

	Args:
		cm (np.array): unsorted confusion matrix
		classes (list): unsorted list of classes
		sorted_classes: list of classes in desired sort order

	Returns:
		sorted_cm (np.array): sorted confusion matrix

	"""
	# Array to hold sorted confusion matrix
	sorted_cm = np.empty((len(sorted_classes), len(sorted_classes)))

	# Array to hold sort indices
	sort_idx = np.empty((len(sorted_classes),), dtype=int)

	for i in range(len(sorted_classes)):
		sort_idx[i] = np.argwhere(classes == sorted_classes[i])

	# Sort
	for j in range(len(sorted_classes)):
		for k in range(len(sorted_classes)):
			sorted_cm[j, k] = cm[sort_idx[j], sort_idx[k]]

	return sorted_cm


def plot_phoneme_confusion_matrix(y_true, y_pred, le, label_type, decode_dir):
	""" Plots phoneme confusion matrix

	Args:
		y_true (np.array): true labels, expressed as ints
		y_pred (np.array): predicted labels, expressed as ints
		le (LabelEncoder): label encoder

	Returns:
		none

	"""
	phoneme_true = phone_to_phoneme(le.inverse_transform(y_true), 39)
	phoneme_pred = phone_to_phoneme(le.inverse_transform(y_pred), 39)
	le_phoneme = preprocessing.LabelEncoder()
	phoneme_true = le_phoneme.fit_transform(phoneme_true)
	phoneme_pred = le_phoneme.transform(phoneme_pred)
	plot_confusion_matrix(phoneme_true, phoneme_pred, le_phoneme, label_type, get_phoneme_list(), decode_dir)


def plot_moa_confusion_matrix(y_true, y_pred, le, label_type, decode_dir):
	""" Plots manner of articulation confusion matrix

	Args:
		y_true (np.array): true labels, expressed as ints
		y_pred (np.array): predicted labels, expressed as ints
		le (LabelEncoder): label encoder

	Returns:
		none

	"""
	moa_true = phone_to_moa(le.inverse_transform(y_true))
	moa_pred = phone_to_moa(le.inverse_transform(y_pred))
	le_moa = preprocessing.LabelEncoder()
	moa_true = le_moa.fit_transform(moa_true)
	moa_pred = le_moa.transform(moa_pred)
	plot_confusion_matrix(moa_true, moa_pred, le_moa, label_type, get_moa_list(), decode_dir)


def write_confmat(cm, classes, decode_dir, label_type):
	""" Writes confusion matrix to a txt file

	Args:
		cm (np.array): confusion matrix in terms of counts
		classes (list): list of sorted classes
		decode_dir (str): directory in which to save the conf mat
		label_type (str): label type

	Returns:
		none

	"""

	# Name of file containing confusion matrix
	cm_file = decode_dir + "/" + label_type + "_confmat.txt"

	with open(cm_file, 'w') as f:
		# Label for true class
		for label in classes:
			f.write('\t' + label)
		f.write('\n')

		for i, label in enumerate(classes):
			# Label for predicted class
			f.write(label)
			# Confusion matrix
			for j in range(np.shape(cm)[1]):
				f.write('\t' + str(cm[i, j]))
			f.write('\n')


def plot_confusion_matrix(y_true, y_pred, le, label_type, sort_order, decode_dir):
	""" Plots confusion matrix

	Args:
		y_true (np.array): true labels, expressed as ints
		y_pred (np.array): predicted labels, expressed as ints
		le (LabelEncoder): label encoder
		sort_order (list): order in which to sort the confusion matrix

	Returns:
		none

	"""
	# Calculate accuracy
	correct = np.sum(y_true == y_pred)
	total = len(y_true)
	accuracy = correct/total
	with open(decode_dir+"/"+label_type+"_accuracy.txt", 'w') as f:
		f.write("Accuracy: " + str(accuracy) + "\n")
		f.write("Correct: " + str(correct) + "\n")
		f.write("Total: " + str(total) + "\n")

	# Calculate confusion matrix in terms of absolute counts
	y_true = le.inverse_transform(y_true)
	y_pred = le.inverse_transform(y_pred)
	cm = confusion_matrix(y_true, y_pred, sort_order)

	# Save confusion matrix to txt file
	write_confmat(cm, sort_order, decode_dir, label_type)

	# Convert confusion matrix from absolute counts to proportions
	sorted_cm = cm.astype('float') / np.tile(np.reshape(np.sum(cm, axis=1), (len(cm), 1)), (1, len(cm)))

	# Plot confusion matrix as a heat map
	plt.figure(figsize=(10, 10))
	plt.imshow(sorted_cm)
	plt.title("Percent Correct = {}%".format(round(accuracy*100, 1)))
	plt.xlabel("Predicted Class")
	plt.ylabel("True Class")
	labels_int = np.arange(0, len(sort_order), 1)
	plt.xticks(labels_int, sort_order, rotation=90)
	plt.yticks(labels_int, sort_order)
	plt.colorbar()
	plt.clim(0, 1)

	# Save figure
	fig_file = decode_dir + "/" + label_type + "_confmat.png"
	plt.savefig(fig_file, bbox_inches='tight')


def get_performance_metrics(summary, conf_dict, decode_dir):
	"""

	Args:
		summary (dict): dictionary containing file name, true class
        predicted class, and probability of predicted class
		conf_dict (dict): configuration parameters
		decode_dir (str): directory in which to save decoding results

	Returns:
		none

	"""

	# Get label encoder
	le = get_label_encoder(conf_dict["label_type"])

	# Plot confusion matrix
	if conf_dict["label_type"] == "phone":
		plot_confusion_matrix(summary['y_true'], summary['y_pred'], le, conf_dict["label_type"], get_phone_list(),
							  decode_dir)
		plot_phoneme_confusion_matrix(summary['y_true'], summary['y_pred'], le, "phoneme", decode_dir)
		plot_moa_confusion_matrix(summary['y_true'], summary['y_pred'], le, "moa", decode_dir)
	elif conf_dict["label_type"] == "phoneme":
		plot_phoneme_confusion_matrix(summary['y_true'], summary['y_pred'], le, conf_dict["label_type"], decode_dir)
		plot_moa_confusion_matrix(summary['y_true'], summary['y_pred'], le, "moa", decode_dir)
	elif conf_dict["label_type"] == "moa":
		plot_moa_confusion_matrix(summary['y_true'], summary['y_pred'], le, conf_dict["label_type"], decode_dir)