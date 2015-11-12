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

		# The weights represent the orientation of a (feature vector size - 1)-dimensional hyperplane.
		self.weights = np.ones(data.shape[1])

		# The bias determines the offset of the hyperplane.
		self.bias = 0

		# The hyperplane will be adjusted so that it separates feature vectors into one of two classes.
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
					# Adjusts the weights.
					self.weights += self.learning_rate * error * features

					# Adjusts the bias.
					self.bias += self.learning_rate * error

					# Increments the error counter.
					error_count += 1

			if error_count == 0:
				# Convergence; can no longer be optimized.
				print('Perceptron converged! ({0} iterations)'.format(iteration))

				return

	def predict(self, features):
		# Calculates a value which -- if sufficiently big -- fires the neuron.
		activator = np.sum(np.multiply(self.weights, features)) + self.bias

		if activator > 0:
			return self.get_label(1)
		else:
			return self.get_label(0)

	def get_signal(self, label):
		# Finds the index of label in self.labels.
		return np.where(self.labels == label)[0][0]

	def get_label(self, signal):
		return self.labels[signal]

	def get_hyperplane(self):
		hyperplane = np.append(self.weights, self.bias)

		# Returns the normalized hyperplane.
		return hyperplane / hyperplane.max()