
from __future__ import print_function
from numpy import array, dot
import numpy as np

from utils import convolve


class ConvRBM(object):
    """
    Convolitional Restricted Bolzman Machine
    """

    def __init__(self, v_shape, n_hiddens, w_size=7, learning_rate=.05,
                 random_state=np.random, n_iter=10, verbose=False):
        self.v_shape = v_shape
        self.w_size = w_size
        self.h_shape = (v_shape[0] - w_size + 1, v_shape[1] - w_size + 1)
        self.n_hiddens = n_hiddens
        self.n_iter = n_iter
        self.verbose = verbose
        self.lr = learning_rate
        self.rng = random_state
        self.W = .01 * self.rng.normal(0, 1, (self.n_hiddens,) + (w_size, w_size))
        self.hiddens = self.rng.uniform(size=(self.n_hiddens,) + self.h_shape)
        self.b = np.zeros((self.n_hiddens,) + self.h_shape)
        self.c = np.zeros((self.v_shape[0] - 2*w_size + 2,
                                           self.v_shape[1] - 2*w_size + 2))

    def fit(self, X):
        n_batches = X.shape[0]
        w, h = self.v_shape        
        while True:
            sum_err = 0
            for vi in xrange(n_batches):
                v = X[vi].reshape(v_shape)
                # TODO: finish

    def _gradients(self, v):
        pass


    def _ff(self, mat):
        """
        Flip matrix both - from left to right and from up to down
        """
        return np.fliplr(np.flipud(mat))

        
    def _sample_hiddens(self, v):
        h = np.zeros((self.n_hiddens,) + self.h_shape)
        for k in self.n_hiddens:
            h[k] = np.exp(convolve(v, _ff(self.W[k])) + self.c)
            

            