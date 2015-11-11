#!/usr/bin/python3

"""perceptrontests.py: Tests the perceptron."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
from perceptron import Perceptron


def get_data(index):
	training_data = np.loadtxt('data/training-data-{0}.csv'.format(index), delimiter = ',')
	test_data = np.loadtxt('data/test-data-{0}.csv'.format(index), delimiter = ',')

	# Gets the labels. They're stored in the last column.
	training_labels = training_data[:, training_data.shape[1] - 1]
	test_labels = test_data[:, test_data.shape[1] - 1]

	# ... And now it forgets the last column.
	training_data = training_data[:, 0 : training_data.shape[1] - 1]
	test_data = test_data[:, 0 : test_data.shape[1] - 1]

	return {
		'training_data': training_data,
		'test_data': test_data,
		'training_labels': training_labels,
		'test_labels': test_labels
	}


def get_correct_predictions(perceptron, data, labels):
	data_count = data.shape[0]
	correct_predictions = 0

	for index in range(0, data_count):
		test_data = data[index, :]
		label = labels[index]
		predicted_label = perceptron.predict(test_data)

		if predicted_label == label:
			correct_predictions += 1

	return correct_predictions


def run():
	perceptron = Perceptron()
	data = get_data(1)

	# Trains the perceptron.
	perceptron.train(data['training_data'], data['training_labels'])

	# Determines the accuracy of its predictions.
	correct_predictions = get_correct_predictions(perceptron, data['test_data'], data['test_labels'])
	total_predictions = data['test_data'].shape[0]

	print('Accuracy: {0}/{1}.'.format(correct_predictions, total_predictions))


if __name__ == '__main__':
    run()