#!/usr/bin/python

'''
Learning Machines
Taught by Patrick Hebron at NYU ITP

MNIST metrics and visualization tools.
'''

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import time

def is_jupyter_notebook():
	try:
		shell = get_ipython().__class__.__name__
		if shell == 'ZMQInteractiveShell':
			return True	  # Jupyter notebook or qtconsole
		elif shell == 'TerminalInteractiveShell':
			return False  # Terminal running IPython
		else:
			return False  # Other type (?)
	except NameError:
		return False	  # Probably standard Python interpreter
		
class MnistSupervisedReporter:
	def __init__(self, name, report_freq, buffer_size = 100):
		self.name		  = name
		self.report_freq  = report_freq
		self.error_buffer = buffer_size
		self.is_jupyter	  = is_jupyter_notebook()
		# Setup plotter data:
		self.error_epoch = []
		self.error_train = []
		self.error_valid = []
		# Setup plotter:
		plt.ion()
		self.fig = plt.figure( 1, dpi=80, figsize=(10, 10) )
		self.fig.clf()
		# Add subplots:
		self.error_plot = self.fig.add_subplot( 2, 1, 1 )
		self.histo_plot = self.fig.add_subplot( 2, 1, 2 )
		# Setup error rate subplot:
		self.error_plot.set_xlabel( 'Epoch' )
		self.error_plot.set_ylabel( 'Error' )
		self.error_train_line = Line2D( [], [], color='black', label='Training' )
		self.error_valid_line = Line2D( [], [], color='green', label='Validation' )
		self.error_plot.add_line( self.error_train_line )
		self.error_plot.add_line( self.error_valid_line )
		self.error_plot.set_ylim( 0.0, 1.0 )
		# Add error legend:
		self.error_legend = self.error_plot.legend(loc='upper left', shadow=True)
		# Show plot:
		if not self.is_jupyter:
			plt.show()

	def saveImage(self, filepath):
		self.fig.savefig( filepath, dpi=self.fig.dpi )

	def update(self, epoch, training_error, validation_error, validation_labels, validation_guesses):
		# Update error plotter data:
		if len( self.error_epoch ) == self.error_buffer:
			self.error_epoch.pop( 0 )
			self.error_train.pop( 0 )
			self.error_valid.pop( 0 )
		self.error_epoch.append( epoch )
		self.error_train.append( training_error )
		self.error_valid.append( validation_error )
		# Set title:
		title = '%s - Epoch: %d, Training Error: %f, Validation Error: %f' % ( self.name, epoch + 1, training_error, validation_error )
		self.error_plot.set_title( title )
		# Compute error plotter x-range:
		mlen = self.report_freq * self.error_buffer
		xmin = np.amin( self.error_epoch )
		xmax = max( xmin + mlen, np.amax( self.error_epoch ) )
		# Update error plotter:
		self.error_train_line.set_data( self.error_epoch, self.error_train )
		self.error_valid_line.set_data( self.error_epoch, self.error_valid )
		self.error_plot.set_xlim( xmin, xmax )
		# Prepare histogram:
		labels_max	  = np.argmax( validation_labels, axis = 1 )
		guesses_max	  = np.argmax( validation_guesses, axis = 1 )
		guess_matches = np.equal( labels_max, guesses_max )
		guess_histo	  = np.histogram( labels_max[ np.where( guess_matches == True ) ], np.arange( 11 ) )[ 0 ].astype( np.float )
		guess_histo	 /= np.histogram( labels_max, np.arange( 11 ) )[ 0 ].astype( np.float )
		histo_rangex  = np.arange( 10 )
		histo_rangey  = np.arange( 11 ) / 10.0
		# Update histogram:
		self.histo_plot.cla()
		self.histo_plot.set_title('Digit Validation Accuracy %')		
		self.histo_plot.set_xticks( histo_rangex + 0.5 )
		self.histo_plot.set_xticklabels( [ str(i) for i in histo_rangex ] )
		self.histo_plot.set_yticks( histo_rangey )
		self.histo_plot.set_yticklabels( [ str(i*100) for i in histo_rangey ] )
		self.histo_plot.bar( histo_rangex, height=guess_histo, width=1 )
		# Draw plot:
		if self.is_jupyter:
			self.fig.canvas.draw()
			time.sleep( 0.01 )
		else:
			plt.draw()
			plt.pause( 0.01 )
			
class MnistUnsupervisedReporter:
	def __init__(self, name, report_freq, buffer_size = 100):
		self.name		  = name
		self.report_freq  = report_freq
		self.error_buffer = buffer_size
		self.is_jupyter	  = is_jupyter_notebook()
		# Setup plotter data:
		self.error_epoch = []
		self.error_train = []
		self.error_valid = []
		# Setup plotter:
		plt.ion()
		self.fig = plt.figure( 1, dpi=80, figsize=(10, 10) )
		self.fig.clf()
		# Add outer grid:
		self.outer_grid = gridspec.GridSpec( 2, 1, wspace=0.2, hspace=0.2 )
		# Add subplots:
		#self.error_plot = self.fig.add_subplot( 1, 1, 1 )
		self.error_plot = plt.subplot( self.outer_grid[0] )
		self.mnist_plot = gridspec.GridSpecFromSubplotSpec( 2, 10, subplot_spec=self.outer_grid[1], hspace=0.05, wspace=0.05 )	 
		# Setup error rate subplot:
		self.error_plot.set_xlabel( 'Epoch' )
		self.error_plot.set_ylabel( 'Error' )
		self.error_train_line = Line2D( [], [], color='black', label='Training' )
		self.error_valid_line = Line2D( [], [], color='green', label='Validation' )
		self.error_plot.add_line( self.error_train_line )
		self.error_plot.add_line( self.error_valid_line )
		self.error_plot.set_ylim( 0.0, 1.0 )
		# Add error legend:
		self.error_legend = self.error_plot.legend(loc='upper left', shadow=True)
		# Initialize MNIST images:
		self.mnist_images = []
		for i in range( 20 ):
			arr = np.zeros( (28, 28) )	
			ax = self.fig.add_subplot( self.mnist_plot[i] )
			self.mnist_images.append( ax.imshow( arr, cmap='gray', vmin=0, vmax=1, aspect='equal', animated=True ) )
			ax.set_xticks([])
			ax.set_yticks([])
			self.fig.add_subplot(ax)
		# Show plot:
		if not self.is_jupyter:
			plt.show()
		
	def saveImage(self, filepath):
		self.fig.savefig( filepath, dpi=self.fig.dpi )

	def update(self, epoch, training_error, validation_error, validation_digits, validation_recons):
		# Update error plotter data:
		if len( self.error_epoch ) == self.error_buffer:
			self.error_epoch.pop( 0 )
			self.error_train.pop( 0 )
			self.error_valid.pop( 0 )
		self.error_epoch.append( epoch )
		self.error_train.append( training_error )
		self.error_valid.append( validation_error )
		# Set title:
		title = '%s - Epoch: %d, Training Error: %f, Validation Error: %f' % ( self.name, epoch + 1, training_error, validation_error )
		self.error_plot.set_title( title )
		# Compute error plotter x-range:
		mlen = self.report_freq * self.error_buffer
		xmin = np.amin( self.error_epoch )
		xmax = max( xmin + mlen, np.amax( self.error_epoch ) )
		# Update error plotter:
		self.error_train_line.set_data( self.error_epoch, self.error_train )
		self.error_valid_line.set_data( self.error_epoch, self.error_valid )
		self.error_plot.set_xlim( xmin, xmax )
		# Update MNIST images:
		vis_idxs = np.random.choice( np.arange( validation_digits.shape[0] ), 10, replace=False )
		vis_digits = validation_digits[ vis_idxs ]
		vis_recons = validation_recons[ vis_idxs ]		
		for i in range( 10 ):
			self.mnist_images[ i	  ].set_data( vis_digits[ i, : ].reshape(28, 28) )
			self.mnist_images[ i + 10 ].set_data( vis_recons[ i, : ].reshape(28, 28) )
		# Draw plot:
		if self.is_jupyter:
			self.fig.canvas.draw()
			time.sleep( 0.01 )
		else:
			plt.draw()
			plt.pause( 0.01 )
