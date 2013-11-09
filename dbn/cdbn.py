
"""Convolutional RBM
"""

# Main author: Andrei Zhabinski
# Based on: rbm.py
# License: BSD Style.

from __future__ import print_function
from operator import mul, sub
from numpy import array, dot
from numpy.random import RandomState
import numpy as np
import matplotlib.pylab as plt
import cv2

from sklearn import linear_model, svm, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.utils import check_arrays
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import safe_sparse_dot as sdot
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
        self.lr = learning_rate
        self.rng = random_state
        self.weights = random_uniform((self.n_hiddens,) + (w_size, w_size))
        self.hiddens = random_uniform((self.n_hiddens,) + self.h_shape)
        self.h_intercepts = random_uniform((self.n_hiddens,) + self.h_shape)
        self.v_intercept = random_uniform((self.v_shape[0] - 2*w_size + 2,
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
            self.weights += self.lr * Grad0[k] - Grad1[k]
        return self._net_probability()


    def fit(X, Y=None):
        for n in self.n_iter:
            for x in X:
                prob = self._fit(x.reshape(self.v_shape))
                if self.verbose:
                    print('Pseudo-likelihood is %d' % prob)
        return self
        
        # V1 = logistic_sigmoid(np.sum())

    def _net_probability():
        """
        Computes pseudo probability of the current network
        """
        # TODO: finish 
        return 1
        
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

    # def _probs_hiddens(self, v):
    #     h_probs = zeros(self.hiddens.shape[0])
    #     for i in range(len(h_probs)):
    #         h_probs[i] = logistic_sigmoid()

        
    # def _mean_hiddens(self, v):
    #     """Computes the probabilities P(h=1|v).

    #     Parameters
    #     ----------
    #     v : array-like, shape (n_samples, n_features)
    #         Values of the visible layer.

    #     Returns
    #     -------
    #     h : array-like, shape (n_samples, n_components)
    #         Corresponding mean field values for the hidden layer.
    #     """
    #     return logistic_sigmoid(safe_sparse_dot(v, self.components_.T)
    #                             + self.intercept_hidden_)

    # def _sample_hiddens(self, v, rng):
    #     """Sample from the distribution P(h|v).

    #     Parameters
    #     ----------
    #     v : array-like, shape (n_samples, n_features)
    #         Values of the visible layer to sample from.

    #     rng : RandomState
    #         Random number generator to use.

    #     Returns
    #     -------
    #     h : array-like, shape (n_samples, n_components)
    #         Values of the hidden layer.
    #     """
    #     p = self._mean_hiddens(v)
    #     p[rng.uniform(size=p.shape) < p] = 1.
    #     return np.floor(p, p)

    # def _sample_visibles(self, h, rng):
    #     """Sample from the distribution P(v|h).

    #     Parameters
    #     ----------
    #     h : array-like, shape (n_samples, n_components)
    #         Values of the hidden layer to sample from.

    #     rng : RandomState
    #         Random number generator to use.

    #     Returns
    #     -------
    #     v : array-like, shape (n_samples, n_features)
    #         Values of the visible layer.
    #     """
    #     p = logistic_sigmoid(np.dot(h, self.components_)
    #                          + self.intercept_visible_)
    #     p[rng.uniform(size=p.shape) < p] = 1.
    #     return np.floor(p, p)

    # def _free_energy(self, v):
    #     """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

    #     Parameters
    #     ----------
    #     v : array-like, shape (n_samples, n_features)
    #         Values of the visible layer.

    #     Returns
    #     -------
    #     free_energy : array-like, shape (n_samples,)
    #         The value of the free energy.
    #     """
    #     return (- safe_sparse_dot(v, self.intercept_visible_)
    #             - np.log(1. + np.exp(safe_sparse_dot(v, self.components_.T)
    #                                  + self.intercept_hidden_)).sum(axis=1))

    # def gibbs(self, v):
    #     """Perform one Gibbs sampling step.

    #     Parameters
    #     ----------
    #     v : array-like, shape (n_samples, n_features)
    #         Values of the visible layer to start from.

    #     Returns
    #     -------
    #     v_new : array-like, shape (n_samples, n_features)
    #         Values of the visible layer after one Gibbs step.
    #     """
    #     rng = check_random_state(self.random_state)
    #     h_ = self._sample_hiddens(v, rng)
    #     v_ = self._sample_visibles(h_, rng)

    #     return v_

    # def _fit(self, v_pos, rng):
    #     """Inner fit for one mini-batch.

    #     Adjust the parameters to maximize the likelihood of v using
    #     Stochastic Maximum Likelihood (SML).

    #     Parameters
    #     ----------
    #     v_pos : array-like, shape (n_samples, n_features)
    #         The data to use for training.

    #     rng : RandomState
    #         Random number generator to use for sampling.

    #     Returns
    #     -------
    #     pseudo_likelihood : array-like, shape (n_samples,)
    #         If verbose=True, pseudo-likelihood estimate for this batch.
    #     """
    #     h_pos = self._mean_hiddens(v_pos)
    #     v_neg = self._sample_visibles(self.h_samples_, rng)
    #     h_neg = self._mean_hiddens(v_neg)

    #     lr = float(self.learning_rate) / v_pos.shape[0]
    #     update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
    #     update -= np.dot(v_neg.T, h_neg).T
    #     self.components_ += lr * update
    #     self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
    #     self.intercept_visible_ += lr * (np.asarray(
    #                                      v_pos.sum(axis=0)).squeeze() -
    #                                      v_neg.sum(axis=0))

    #     h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
    #     self.h_samples_ = np.floor(h_neg, h_neg)

    #     if self.verbose:
    #         return self.score_samples(v_pos)

    # def score_samples(self, v):
    #     """Compute the pseudo-likelihood of v.

    #     Parameters
    #     ----------
    #     v : {array-like, sparse matrix} shape (n_samples, n_features)
    #         Values of the visible layer.

    #     Returns
    #     -------
    #     pseudo_likelihood : array-like, shape (n_samples,)
    #         Value of the pseudo-likelihood (proxy to likelihood).
    #     """
    #     rng = check_random_state(self.random_state)
    #     fe = self._free_energy(v)

    #     if issparse(v):
    #         v_ = v.toarray()
    #     else:
    #         v_ = v.copy()
    #     i_ = rng.randint(0, v.shape[1], v.shape[0])
    #     v_[np.arange(v.shape[0]), i_] = 1 - v_[np.arange(v.shape[0]), i_]
    #     fe_ = self._free_energy(v_)

    #     return v.shape[1] * logistic_sigmoid(fe_ - fe, log=True)

    # def fit(self, X, y=None):
    #     """Fit the model to the data X.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix} shape (n_samples, n_features)
    #         Training data.

    #     Returns
    #     -------
    #     self : BernoulliRBM
    #         The fitted model.
    #     """
    #     X, = check_arrays(X, sparse_format='csr', dtype=np.float)
    #     n_samples = X.shape[0]
    #     rng = check_random_state(self.random_state)

    #     self.components_ = np.asarray(
    #         rng.normal(0, 0.01, (self.n_components, X.shape[1])),
    #         order='fortran')
    #     self.intercept_hidden_ = np.zeros(self.n_components, )
    #     self.intercept_visible_ = np.zeros(X.shape[1], )
    #     self.h_samples_ = np.zeros((self.batch_size, self.n_components))

    #     n_batches = int(np.ceil(float(n_samples) / self.batch_size))
    #     batch_slices = list(gen_even_slices(n_batches * self.batch_size,
    #                                         n_batches, n_samples))
    #     verbose = self.verbose
    #     for iteration in xrange(self.n_iter):
    #         pl = 0.
    #         if verbose:
    #             begin = time.time()

    #         for batch_slice in batch_slices:
    #             pl_batch = self._fit(X[batch_slice], rng)

    #             if verbose:
    #                 pl += pl_batch.sum()

    #         if verbose:
    #             pl /= n_samples
    #             end = time.time()
    #             print("Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
    #                   % (iteration, pl, end - begin))

    #     return self


def run():
    im = cv2.imread(list_images('../data/gender/female')[2])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.equalizeHist(im)
    im = cv2.resize(im, (96, 96))
    kernels = gabor_kernels()
    filtered = [nd.convolve(im, k, mode='wrap') for k in kernels]
    smartshow(filtered)
    return filtered


def run2():
    V0 = random_uniform(shape=(96, 96))
    crbm = ConvolutionalRBM((96, 96), 10)
    crbm._fit(V0)

