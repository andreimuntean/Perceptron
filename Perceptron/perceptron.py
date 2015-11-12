#!/usr/bin/python3

"""perceptron.py: Predicts whether some data belongs to one class or another."""

__author__ = 'Andrei Muntean'
__license__ = 'MIT License'

import numpy as np


class Perceptron:
	def __init__(self, learning_rate = 1):
		self.learning_rate = learning_rate

	def train(self, data, labels, maximum_iterations = 50):
		# Stores the labels.
		self.labels = np.unique(labels)

		# Initializes the weights.
		self.weights = np.zeros(data.shape[1])

		# Initializes the threshold.
		self.threshold = 0

		for iteration in range(1, maximum_iterations):
			error_count = 0;

			# Goes through every example.
			for index in range(1, data.shape[0]):
				features = data[index, :]

				# Predicts a label.
				label = self.predict(features)

				# Determines whether the prediction is incorrect.
				error = self.get_signal(labels[index]) - self.get_signal(label)

				if not error == 0:
					# Updates the weights.
					self.weights = self.weights + self.learning_rate * error * features

					# Updates the threshold.
					self.threshold = self.threshold + self.learning_rate * error * -1

					# Increments the error counter.
					error_count += 1

			if error_count == 0:
				# Convergence; can no longer be optimized.
				print('Perceptron converged! ({0} iterations)'.format(iteration))

				return

	def predict(self, features):
		# Calculates the value that will be compared with the threshold.
		activator = np.sum(np.multiply(self.weights, features))

		if activator > self.threshold:
			return self.get_label(1)
		else:
			return self.get_label(0)

	def get_signal(self, label):
		# Finds the index of label in self.labels.
		return np.where(self.labels == label)[0][0]

	def get_label(self, signal):
		return self.labels[signal]