#!/usr/bin/python

'''
Learning Machines
Taught by Patrick Hebron at NYU ITP

Multilayer Perceptron (MLP) implementation.
'''

import numpy as np
import Activation
from MnistReporter import *

class Mlp:

	class Layer:
		def __init__(self,input_size,output_size):
			wlimit = np.sqrt( 6.0 / float( input_size + output_size ) )
			self.weights = np.random.uniform( -wlimit, wlimit, ( input_size, output_size ) )
			self.bias	 = np.zeros( output_size )
		
	def __init__(self, name, layer_sizes, activation_fn_name):
		self.name = name
		# Create layers:
		self.layers = []
		for i in range( len( layer_sizes ) - 1 ):
			self.layers.append( self.Layer( layer_sizes[ i ], layer_sizes[ i + 1 ] ) )
		# Set activation function:
		if activation_fn_name == 'tanh':
			self.activation_fn	= Activation.tanh
			self.activation_dfn = Activation.dtanh
		else:
			self.activation_fn	= Activation.sigmoid
			self.activation_dfn = Activation.dsigmoid

	def getErrorRate(self, labels, guesses):
		'''returns mean square error'''
		return np.mean( np.square( labels - guesses ) )

	def predictSignal(self, input):
		# Setup signals:
		activations = [ input ]
		outputs		= [ input ]
		# Feed forward through layers:
		for i in range( 1, len( self.layers ) + 1 ):	
			# Compute activations:
			curr_act = np.dot( outputs[ i - 1 ], self.layers[ i - 1 ].weights ) + self.layers[ i - 1 ].bias
			# Append current signals:
			activations.append( curr_act )
			outputs.append( self.activation_fn( curr_act ) )
		# Return signals:
		return activations, outputs

	def predict(self, input):
		# Feed forward:
		activations, outputs = self.predictSignal( input )
		# Return final layer output:
		return outputs[ -1 ]

	def trainEpoch(self, training_samples, training_labels, learn_rate, batch_size):
		error	   = 0.0
		num_rows   = training_samples.shape[ 0 ]
		num_layers = len( self.layers )
		# Iterate over each training batch:
		for bstart in range( 0, num_rows, batch_size ):
			# Compute batch stop index:
			bstop = min( bstart + batch_size, num_rows )
			# Compute batch size:
			bsize = bstop - bstart
			# Compute batch multiplier:
			bmult = learn_rate * ( 1.0 / float( bsize ) )
			# Slice data:
			bsamples = training_samples[ bstart:bstop, : ]
			blabels	 = training_labels[ bstart:bstop, : ]
			# Feed forward:
			bactivations, boutputs = self.predictSignal( bsamples )
			# Prepare batch deltas:
			bdeltas = []
			# Back propagate from final outputs:
			bdeltas.append( self.activation_dfn( bactivations[ num_layers ] ) * ( boutputs[ num_layers ] - blabels ) )
			# Back propagate remaining layers:
			for i in range( num_layers - 1, 0, -1 ):			
				bdeltas.append( self.activation_dfn( bactivations[ i ] ) * np.dot( bdeltas[ -1 ], self.layers[ i ].weights.T ) )
			# Apply batch deltas:
			for i in range( num_layers ):			
				self.layers[ i ].weights -= bmult * np.dot( boutputs[ i ].T, bdeltas[ num_layers - i - 1 ] )
				self.layers[ i ].bias	 -= bmult * np.sum( bdeltas[ num_layers - i - 1 ], axis = 0 )
			# Scale batch error and accumulate total:
			error += self.getErrorRate( blabels, boutputs[ -1 ] ) * ( float( bsize ) / float( num_rows ) )
		# Return training error:
		return error
				
	def train(self, training_samples, training_labels, validation_samples, validation_labels, learn_rate, epochs, batch_size = 10, report_freq = 10, report_buff = 100):
		 # Setup error reporter:
		error_reporter = MnistSupervisedReporter( self.name, report_freq, report_buff )
		# Iterate over each training epoch:
		for epoch in range( epochs ):
			# Perform training:
			training_error = self.trainEpoch( training_samples, training_labels, learn_rate, batch_size )
			# Report error, if applicable:
			if ( epoch + 1 ) % report_freq == 0:
				# Compute validation error:
				validation_guesses = self.predict( validation_samples )
				validation_error = self.getErrorRate( validation_labels, validation_guesses )
				# Update error reporter:
				error_reporter.update( epoch, training_error, validation_error, validation_labels, validation_guesses )
		# Save final training visualization to image:
		error_reporter.saveImage( 'report_' + self.name + '_training.png' )
