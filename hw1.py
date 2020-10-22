"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.linear_layers = [Linear(x[0], x[1], weight_init_fn,bias_init_fn) for x in zip([input_size] + hiddens,hiddens + [output_size])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(x) for x in hiddens[:num_bn_layers]]



    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        y = x
        for i,layer in enumerate(self.linear_layers):
            z = layer(y)
            if self.bn:
                try:
                    bn_layer = self.bn_layers[i]
                    z = bn_layer(z, eval = not self.train_mode)
                except IndexError:
                    pass
            y = self.activations[i](z)
        self.output = y
        return self.output

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            layer = self.linear_layers[i]
            layer.dW.fill(0.0)
            layer.db.fill(0.0)
        
        if self.bn:
            for i in range(len(self.bn_layers)):
                layer = self.bn_layers[i]
                layer.dgamma.fill(0.0)
                layer.dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            layer = self.linear_layers[i]
            dW = self.momentum * layer.momentum_W - self.lr * layer.dW
            layer.momentum_W = dW
            layer.W += dW
            db = self.momentum * layer.momentum_b - self.lr * layer.db
            layer.momentum_b = db
            layer.b += db
        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                layer = self.bn_layers[i]
                layer.gamma -= self.lr * layer.dgamma
                layer.beta -= self.lr * layer.dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.total_loss(labels)
        delta = self.criterion.derivative()
        for i in reversed(range(len(self.linear_layers))):
            if i != (len(self.linear_layers)):
                delta = self.activations[i].derivative() * delta
            if self.bn:
                try:
                    bn_layer = self.bn_layers[i]
                    delta = bn_layer.backward(delta)
                except IndexError:
                    pass
            delta = self.linear_layers[i].backward(delta)

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)


    for e in range(nepochs):
        np.random.shuffle(idxs)
        
        trainx = trainx[idxs]
        trainy = trainy[idxs]
        
        training_loss = 0
        training_error = 0
        training_n = 0
        
        for b in range(0, len(trainx), batch_size):
            mlp.train()
            mlp.zero_grads()
            
            x = trainx[b:b+batch_size]
            y = trainy[b:b+batch_size]
            
            mlp.forward(x)
            training_loss += mlp.total_loss(y)
            training_error += mlp.error(y)
            training_n += batch_size
            mlp.backward(y)
            mlp.step()
            
        
        training_losses[e] = training_loss/training_n
        training_errors[e] = training_error/training_n
        
        validation_loss = 0
        validation_error = 0
        validation_n = 0
        
        for b in range(0, len(valx), batch_size):
            mlp.eval()
            x = valx[b:b+batch_size]
            y = valy[b:b+batch_size]
            mlp.forward(x)
            validation_loss += mlp.total_loss(y)
            validation_error += mlp.error(y)
            validation_n += batch_size
        
        validation_losses[e] = validation_loss/validation_n
        validation_errors[e] = validation_error/validation_n
        

    return (training_losses, training_errors, validation_losses, validation_errors)

