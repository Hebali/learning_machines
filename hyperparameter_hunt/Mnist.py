#!/usr/bin/python

'''
Learning Machines
Taught by Patrick Hebron at NYU ITP

MNIST dataset helpers.
'''

import os
import gzip
import pickle
import numpy as np

def mnist_encode_one_hot_label(idx):
	'''MNIST one-hot encoder function'''
	enc = np.zeros( 10 )
	enc[ idx ] = 1.0
	return enc

def mnist_decode_one_hot_label(enc):
	'''MNIST one-hot decoder function'''
	return np.argmax( enc )

def mnist_get_accuracy(labels, guesses):
	'''returns percentage of MNIST guesses that match labels'''
	return np.mean( np.equal( np.argmax( labels, axis = 1 ), np.argmax( guesses, axis = 1 ) ).astype( np.float64 ) )

class Mnist:
	def __init__(self, threshold = True):
		# Set pickle path:
		mnist_pickle_path = 'mnist.pkl.gz'
		# Download pickle, if necessary:
		if not os.path.exists( mnist_pickle_path ):
			# fix for python3 compatibility. credit: https://stackoverflow.com/a/3969809/5420567
			import urllib.request
			downloader = urllib.request.URLopener()
			downloader.retrieve( 'http://deeplearning.net/data/mnist/mnist.pkl.gz', mnist_pickle_path )

		# fix for python3 compatibility. credit: http://www.mlblog.net/2016/09/reading-mnist-in-python3.html
		# Load pickle:
		with gzip.open(mnist_pickle_path, 'rb') as fh:
			u = pickle._Unpickler(fh)
			u.encoding = 'latin1'
			training_data, validation_data, testing_data = u.load()

		# Format dataset:
		self.training_digits, self.training_labels = self.format_dataset( training_data, threshold )
		self.validation_digits, self.validation_labels = self.format_dataset( validation_data, threshold )
		self.testing_digits, self.testing_labels = self.format_dataset( testing_data, threshold )

	def getTrainingData(self, count = 0):
		if count == 0:
			return ( self.training_digits, self.training_labels )
		else:
			return self.get_batch( count, self.training_digits, self.training_labels )

	def getValidationData(self, count = 0):
		if count == 0:
			return ( self.validation_digits, self.validation_labels )
		else:
			return self.get_batch( count, self.validation_digits, self.validation_labels )

	def getTestingData(self, count = 0):
		if count == 0:
			return ( self.testing_digits, self.testing_labels )
		else:
			return self.get_batch( count, self.testing_digits, self.testing_labels )

	@staticmethod
	def get_batch(count,digits,labels):
		total = len( digits )
		count = min( count, total )
		idxs  = np.random.choice( np.arange( total ), count, replace=False )
		return ( digits[ idxs ], labels[ idxs ] )

	@staticmethod
	def format_dataset(dataset, threshold):
		digits = np.array( [ np.reshape( x, 784 ) for x in dataset[ 0 ] ] )
		labels = np.array( [ mnist_encode_one_hot_label( y ) for y in dataset[ 1 ] ] )
		return ( ( digits > 0 ).astype( np.float ) if threshold else digits, labels )
