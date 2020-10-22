# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10) 
            y (np.array): (batch size, 10) 
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        x_shift = x - np.max(x, axis = 1, keepdims = True)
        x_exp = np.exp(x_shift)
        self.probs = x_exp/np.sum(x_exp, axis = 1, keepdims = True)
        return np.sum(np.multiply(y, np.log(self.probs))*(-1),axis = 1)

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.probs - self.labels
