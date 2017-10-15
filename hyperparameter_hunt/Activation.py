#!/usr/bin/python

'''
Learning Machines
Taught by Patrick Hebron at NYU ITP

Activation function implementations.
'''

import numpy as np

def sigmoid(x):
	'''sigmoid function'''
	return 1.0 / ( 1.0 + np.exp( -x ) )

def dsigmoid(x):
	'''sigmoid derivative function'''
	y = sigmoid( x )
	return y * ( 1.0 - y )

def tanh(x):
	'''tanh function'''
	return np.sinh( x ) / np.cosh( x )

def dtanh(x):
	'''tanh derivative function'''
	return 1.0 - np.power( tanh( x ), 2.0 )
