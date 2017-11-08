#!/usr/bin/python

'''
Learning Machines
Taught by Patrick Hebron at NYU ITP

Example code for The Great Hyperparameter Hunt assignment.
'''

from Mnist import *
from Supervised import *
from Unsupervised import *
import numpy as np
np.set_printoptions( precision = 3, suppress = True )


'''
NOTE:

The hyperparameter configurations below are for demonstration purposes only.
Your job is to find better ones.

Please see README for more information on this assignment.
'''


# General settings (you CAN change these):
mnist_use_threshold = False

# RBM hyperparameters (you CAN change these):
rbm_is_continuous	= True
rbm_visible_size	= 784
rbm_hidden_size		= 50
rbm_batch_size		= 10
rbm_learn_rate		= 0.01
rbm_cd_steps		= 1
rbm_training_epochs = 100
rbm_report_freq		= 1
rbm_report_buffer	= rbm_training_epochs

# MLP hyperparameters (you CAN change these):
mlp_layer_sizes		= [ rbm_hidden_size, 20, 10 ]
mlp_batch_size		= 10
mlp_learn_rate		= 0.05
mlp_training_epochs = 100
mlp_report_freq		= 1
mlp_report_buffer	= mlp_training_epochs

# MNIST training example counts (you CANNOT change these):
mnist_num_training_examples	  = 10000
mnist_num_validation_examples =	 5000
mnist_num_testing_examples	  =	 5000

# Load MNIST dataset:
mnist = Mnist( mnist_use_threshold )

training_digits,   training_labels	 = mnist.getTrainingData( mnist_num_training_examples )
validation_digits, validation_labels = mnist.getValidationData( mnist_num_validation_examples )
testing_digits,	   testing_labels	 = mnist.getTestingData( mnist_num_testing_examples )

# Initialize and train RBM:
rbm_name = 'rbm_' + str(rbm_visible_size) + '_' + str(rbm_hidden_size)
rbm = Rbm( rbm_name, rbm_visible_size, rbm_hidden_size, rbm_is_continuous )
rbm.train( training_digits, validation_digits, rbm_learn_rate, rbm_cd_steps, rbm_training_epochs, rbm_batch_size, rbm_report_freq, rbm_report_buffer )

# Encode datasets with RBM:
_, training_encodings = rbm.getHiddenSample( training_digits )
_, validation_encodings = rbm.getHiddenSample( validation_digits )
_, testing_encodings = rbm.getHiddenSample( testing_digits )

# Initialize and train MLP:
mlp_name = 'mlp_' + '_'.join( str(i) for i in mlp_layer_sizes )
mlp = Mlp( mlp_name, mlp_layer_sizes, 'sigmoid' )
mlp.train( training_encodings, training_labels, validation_encodings, validation_labels, mlp_learn_rate, mlp_training_epochs, mlp_batch_size, mlp_report_freq, mlp_report_buffer )

# Perform final test:
testing_guesses = mlp.predict( testing_encodings )
testing_error = mlp.getErrorRate( testing_labels, testing_guesses )
testing_accuracy = mnist_get_accuracy( testing_labels, testing_guesses )
print ('Final Testing Error Rate: %f' % ( testing_error ))
print ('Final Testing Accuracy: %f' % ( testing_accuracy ))
