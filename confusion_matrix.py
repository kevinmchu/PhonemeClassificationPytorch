# confusionMatrix.py
# Author: Kevin Chu
# Last Modified: 05/11/2020

import matplotlib.pyplot as plt
import numpy as np
from phone_mapping import phone_to_phoneme
from phone_mapping import phone_to_moa
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


def plot_phoneme_confusion_matrix(y_true, y_pred, le):
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
	plot_confusion_matrix(phoneme_true, phoneme_pred, le_phoneme, get_phoneme_list())


def plot_moa_confusion_matrix(y_true, y_pred, le):
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
	plot_confusion_matrix(moa_true, moa_pred, le_moa, get_moa_list())


def plot_confusion_matrix(y_true, y_pred, le, sort_order):
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
	accuracy = float(np.sum(y_true == y_pred))/len(y_true)

	# Calculate normalized confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / np.tile(np.reshape(np.sum(cm, axis=1), (len(cm), 1)), (1, len(cm)))

	# Labels
	labels_int = np.arange(0, len(np.unique(y_true)), 1)
	labels_str = le.inverse_transform(np.unique(y_true))

	# Sort
	sorted_cm = sort_classes(cm, labels_str, sort_order)

	plt.figure(figsize=(10, 10))
	plt.imshow(sorted_cm)
	plt.title("Percent Correct = {}%".format(round(accuracy*100, 1)))
	plt.xlabel("Predicted Class")
	plt.ylabel("True Class")
	plt.xticks(labels_int, sort_order, rotation=90)
	plt.yticks(labels_int, sort_order)
	plt.colorbar()
	plt.clim(0, 1)
	plt.show()
	#plt.savefig("fig.png", bbox_inches='tight')
