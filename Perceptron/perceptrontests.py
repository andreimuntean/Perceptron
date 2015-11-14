#!/usr/bin/python3

"""perceptrontests.py: Tests the perceptron."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


def get_data(index):
	training_data = np.loadtxt('data/training-data-{0}.csv'.format(index), delimiter = ',')
	test_data = np.loadtxt('data/test-data-{0}.csv'.format(index), delimiter = ',')

	# Gets the labels. They're stored in the first column.
	training_labels = training_data[:, 0]
	test_labels = test_data[:, 0]

	# ... And now it forgets the first column.
	training_data = training_data[:, 1:]
	test_data = test_data[:, 1:]

	return {
		'training_data': training_data,
		'test_data': test_data,
		'training_labels': training_labels,
		'test_labels': test_labels
	}


def get_predictions(perceptron, data):
	data_count = data.shape[0]
	predictions = np.empty([data_count])

	for index in range(0, data_count):
		test_data = data[index, :]
		predictions[index] = perceptron.predict(test_data)

	return predictions


def count_correct_predictions(actual_labels, predicted_labels):
	predicted_correctly = (predicted_labels == actual_labels)

	# Removes all false values.
	predicted_correctly = predicted_correctly[predicted_correctly]
	
	# Counts the number of correct predictions.
	return predicted_correctly.shape[0]


def split_into_classes(data, labels):
	data = np.concatenate((labels[:, None], data), 1)
	first_class = data[data[:, 0] == 0]
	second_class = data[data[:, 0] == 1]

	return [first_class[:, 1:], second_class[:, 1:]]


def plot_2d_results(perceptron, data):
	"""Only works if the feature vector is bidimensional."""

	# Divides the data into classes.
	training_data_classes = split_into_classes(data['training_data'], data['training_labels'])
	test_data_classes = split_into_classes(data['test_data'], data['test_labels'])

	# Plots the data.
	plt.plot(training_data_classes[0][:, 0], training_data_classes[0][:, 1], 'bo',
		training_data_classes[1][:, 0], training_data_classes[1][:, 1], 'ro',
		test_data_classes[0][:, 0], test_data_classes[0][:, 1], 'b*',
		test_data_classes[1][:, 0], test_data_classes[1][:, 1], 'r*',
		markersize = 12)

	# Constructs a line that represents the decision boundary.
	weights = perceptron.weights
	bias = perceptron.bias
	x_range = np.array([0, 100])
	y_range = -(x_range * weights[0] + bias) / weights[1]

	# Plots the decision boundary.
	plt.plot(x_range, y_range, 'k')
	plt.show()


def run():
	perceptron = Perceptron()
	data = get_data(2)

	# Trains the perceptron.
	perceptron.train(data['training_data'], data['training_labels'])

	# Tests the perceptron.
	predictions = get_predictions(perceptron, data['test_data'])
	correct_prediction_count = count_correct_predictions(data['test_labels'], predictions)

	# Displays the results.
	print('Accuracy: {0}/{1}.'.format(correct_prediction_count, predictions.shape[0]))
	print('Hyperplane: {0}'.format(np.append(perceptron.weights, perceptron.bias)))
	plot_2d_results(perceptron, data)


if __name__ == '__main__':
    run()