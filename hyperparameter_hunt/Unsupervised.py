#!/usr/bin/python

'''
Learning Machines
Taught by Patrick Hebron at NYU ITP

Restricted Boltzmann Machine (RBM) implementation.
'''

import numpy as np
import Activation
from MnistReporter import *

class Rbm:
	def __init__(self, name, sizeV, sizeH, continuous = False):
		self.name	 = name
		self.is_crbm = continuous
		# Initialize weights:
		self.weights = np.array( np.random.uniform( -1.0 / sizeV, 1.0 / sizeV, ( sizeV, sizeH ) ) )
		# Initialize biases:
		self.biasH	 = np.zeros( sizeH )
		self.biasV	 = np.zeros( sizeV )

	def getErrorRate(self, samples, reconstructions):
		'''returns mean square error'''
		return np.mean( np.square( samples - reconstructions ) )
		
	def trainEpoch(self, training_samples, learn_rate, cd_steps, batch_size):
		error	   = 0.0
		num_rows   = training_samples.shape[ 0 ]
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
			# Get hidden activations and samples:
			aH_0, sH_0 = self.getHiddenSample( bsamples )
			# Perform each contrastive divergence step:
			for i in range( cd_steps ):
				aV_inf, sV_inf, aH_inf, sH_inf = self.getGibbsHvh( ( sH_0 if i == 0 else sH_inf ) )
			# Update weights:
			self.weights += bmult * ( np.dot( bsamples.T, aH_0 ) - np.dot( sV_inf.T, aH_inf ) )
			# Update biases:
			self.biasV	 += bmult * np.mean( bsamples - sV_inf, axis = 0 )
			self.biasH	 += bmult * np.mean( aH_0 - aH_inf, axis = 0 )
			# Scale batch error and accumulate total:
			error += self.getErrorRate( bsamples, sV_inf ) * ( float( bsize ) / float( num_rows ) )
		# Return training error:
		return error

	def train(self, training_samples, validation_samples, learn_rate, cd_steps, epochs, batch_size = 10, report_freq = 10, report_buff = 100):
		# Setup error reporter:
		error_reporter = MnistUnsupervisedReporter( self.name, report_freq, report_buff )
		# Perform each training epoch:
		for epoch in range( epochs ):
			training_error = self.trainEpoch( training_samples, learn_rate, cd_steps, batch_size )
			# Report error, if applicable:
			if ( epoch + 1 ) % report_freq == 0:
				# Compute validation error:
				val_aH, val_sH, val_aV, val_sV = self.getGibbsVhv( validation_samples )
				validation_error = self.getErrorRate( validation_samples, val_sV )
				# Update error reporter:
				error_reporter.update( epoch, training_error, validation_error, validation_samples, val_sV )
		# Save final training visualization to image:
		error_reporter.saveImage( 'report_' + self.name + '_training.png' )

	def getHiddenActivations(self, inputV):
		return Activation.sigmoid( np.dot( inputV, self.weights ) + self.biasH )

	def getVisibleActivations(self, inputH):
		return Activation.sigmoid( np.dot( inputH, self.weights.T ) + self.biasV )

	def getHiddenSample(self, inputV, force_binomial = False):
		aH = self.getHiddenActivations( inputV )
		return [ aH, self.getSample( aH ) if ( force_binomial or not self.is_crbm ) else aH ]

	def getVisibleSample(self, inputH, force_binomial = False):
		aV = self.getVisibleActivations( inputH )
		return [ aV, self.getSample( aV ) if ( force_binomial or not self.is_crbm ) else aV ]

	def getGibbsHvh(self, inputH, force_binomial = False):
		aV, sV = self.getVisibleSample( inputH, force_binomial )
		aH, sH = self.getHiddenSample( sV, force_binomial )
		return [ aV, sV, aH, sH ]

	def getGibbsVhv(self, inputV, force_binomial = False):
		aH, sH = self.getHiddenSample( inputV, force_binomial )
		aV, sV = self.getVisibleSample( sH, force_binomial )
		return [ aH, sH, aV, sV ]
		
	@staticmethod
	def getSample(activations):
		return np.random.binomial( 1, activations, activations.shape )
		
		
