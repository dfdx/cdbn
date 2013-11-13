
"""Convolutional RBM
"""

# Main author: Andrei Zhabinski
# Based on: rbm.py
# License: BSD Style.

from __future__ import print_function
from numpy import array, dot
from numpy.random import RandomState
import numpy as np
import matplotlib.pylab as plt

from sklearn import linear_model, svm, datasets, metrics
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils.extmath import logistic_sigmoid

from scipy import ndimage as nd
from skimage.filter import gabor_kernel

from helpers import smartshow, list_images
from utils import random_uniform, convolve




class ConvolutionalRBM(BaseEstimator, TransformerMixin):

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
        self.weights = self.rng.normal(0, .01,
                                             (self.n_hiddens,) + (w_size, w_size))
        self.hiddens = self.rng.uniform(size=(self.n_hiddens,) + self.h_shape)
        self.h_intercepts = np.zeros((self.n_hiddens,) + self.h_shape)
        self.v_intercept = np.zeros((self.v_shape[0] - 2*w_size + 2,
                                           self.v_shape[1] - 2*w_size + 2))
        

    def _fit(self, V0):
        Ph0 = np.zeros((self.n_hiddens,) + self.h_shape)
        H0 = np.zeros((self.n_hiddens,) + self.h_shape)
        Grad0 = np.zeros((self.n_hiddens,) + (self.w_size, self.w_size))
        for k in xrange(self.n_hiddens):          
            Ph0[k] = logistic_sigmoid(convolve(V0, self.weights[k])
                                      + self.h_intercepts[k])
            Grad0[k] = convolve(V0, Ph0[k])
            H0[k][self.rng.uniform(size=self.h_shape) < Ph0[k]] = 1
            
        h_convolved = self.v_intercept
        for k in xrange(self.n_hiddens):
            h_convolved += convolve(H0[k], np.flipud(np.fliplr(self.weights[k])))
        V1m = logistic_sigmoid(h_convolved)
        V1 = V0.copy()
        middle_offset = self.w_size - 1
        V1[middle_offset:-middle_offset, middle_offset:-middle_offset] = V1m
        
        Ph1 = np.zeros((self.n_hiddens,) + self.h_shape)        
        Grad1 = np.zeros((self.n_hiddens,) + (self.w_size, self.w_size))
        for k in xrange(self.n_hiddens):
            Ph1[k] = logistic_sigmoid(convolve(V1, self.weights[k])
                                      + self.h_intercepts[k])
            Grad1[k] = convolve(V1, Ph1[k])
            self.weights += self.lr * (Grad0[k] - Grad1[k])
        return self._net_probability(V0)


    def fit(self, X, Y=None):
        print('n_iter is %s' % self.n_iter)
        for n in xrange(self.n_iter):
            for x in X:
                prob = self._fit(x.reshape(self.v_shape))
            if self.verbose:
                print('Pseudo-likelihood is %s on iteration %d' % (prob, n + 1))
        return self
        
        # V1 = logistic_sigmoid(np.sum())

    def _net_probability(self, V):
        """
        Computes pseudo probability of the current network
        """
        v_energy = 0
        for k in xrange(self.n_hiddens):
            v_energy -= (self.hiddens[k] * convolve(V, self.weights[k])).sum()
        h_int_energy = 0
        for k in xrange(self.n_hiddens):
            h_int_energy -= self.h_intercepts[k].sum() * self.hiddens[k].sum()
        v_int_energy = - self.v_intercept.sum() * V.sum()
        energy = v_energy + h_int_energy + v_int_energy
        print(energy)
        return logistic_sigmoid(- energy)
        
        


        
        
    # def transform(self, X):
    #     """Compute the hidden layer activation probabilities, P(h=1|v=X).

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} shape (n_samples, n_features)
    #         The data to be transformed.

    #     Returns
    #     -------
    #     h : array, shape (n_samples, n_components)
    #         Latent representations of the data.
    #     """
    #     X, = check_arrays(X, sparse_format='csr', dtype=np.float)
    #     return self._mean_hiddens(X)


def run():
    import cv2
    im = cv2.imread(list_images('../data/gender/female')[2])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.equalizeHist(im)
    im = cv2.resize(im, (96, 96))
    kernels = gabor_kernels()
    filtered = [nd.convolve(im, k, mode='wrap') for k in kernels]
    smartshow(filtered)
    return filtered


def run2():
    import cv2
    V0 = random_uniform(shape=(96, 96))
    crbm = ConvolutionalRBM((96, 96), 10)
    crbm._fit(V0)

def run3():
    import cv2
    size = 96
    im_list = list(list_images('../data/gender/female'))
    X = np.zeros((len(im_list), size, size))
    print('Reading images...')
    for i in xrange(len(im_list)):
        im = cv2.imread(im_list[i])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.equalizeHist(im)
        im = cv2.resize(im, (size, size))
        X[i] = im.astype(np.float64) / 256
    print('X.shape is %s' % (X.shape,))
    print('Fitting...')
    crbm = ConvolutionalRBM((size, size), 10, w_size=7, n_iter=3, verbose=True)
    crbm.fit(X)
    return crbm

def run_digits():
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X /= 256
    crbm = ConvolutionalRBM((8, 8), 100, w_size=3, n_iter=3, verbose=True)
    crbm.fit(X)
    return crbm