# [Learning Machines](http://www.patrickhebron.com/learning-machines/index.html)

###### Taught by [Patrick Hebron](http://www.patrickhebron.com) at [NYU/ITP](http://itp.nyu.edu), Fall 2017

****

### The Great Hyperparameter Hunt
****

#### Purpose

The purpose of this assignment is to have you:

* Experiment with the dimensionality reduction capabilities of a Restricted Boltzmann Machine.
* Explore how this unsupervised learning system can be coupled with a supervised learning system.
* Develop the skill of selecting a neural architecture and accompanying hyperparameters for a given learning problem through structured experimentation.

#### Goal

I have provided Multilayer Perceptron (MLP) and Restricted Boltzmann Machine (RBM) implementations as well as some code to load the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and visualize training results.

In *main.py*, I have provided a baseline approach to classifying MNIST digits using a single RBM and MLP. I have intentionally written this file in a clumbsy way...

Notice that after training a single RBM, I then call *rbm.getHiddenSample()* to generate encodings for use in the subsequent training of the MLP.

To improve results, you might try using those encoding to train another RBM that will reduce the data dimensionality even further before training the MLP on the resulting encodings.

You can do this with numerous RBMs (often called *Stacked Autoencoders*).

It may be helpful to implement a StackedRbm class in order to streamline your workflow and speed up your experimentation process. 

Spend some time experimenting with stacked RBMs. Autoencoders are fascinating and powerful tools!

Ultimately, your job is to achieve the highest possible accuracy rate in predicting outputs from the *testing examples* of the MNIST dataset.

We'll look at results next week and treat this as a friendly competition to discover the best accuracy rate.

#### Rules of the Game

You *cannot*:

* Change the number of training/validation/testing examples used.
* Change the MLP or RBM getErrorRate() or the mnist\_get\_accuracy() functions.

You *can*:

* Use any number of RBMs with layer sizes of your choosing.
* Use any number of MLP layers with layer sizes of your choosing.
* Change any configuration setting or hyperparameter in the experiment (except those mentioned in the *cannot* section above).
* Implement a Stacked RBM class if it helps to streamline your workflow.
* Change the MLP and RBM implementations in any way you like so long as those changes do not attempt to undermine the fairness of this competition. e.g. You can implement Dropout or Regularization in the MLP class, but you cannot manipulate the error metrics used to evaluate your results.

