"""
Convolutional Deep Belief Network
Based on: https://github.com/dustinstansbury/medal/blob/master/models/crbm.m
"""

from __future__ import print_function
from numpy import array, dot
import numpy as np
from sklearn.externals.six.moves import xrange
from sklearn.utils.extmath import logistic_sigmoid
from scipy import signal
from utils import conv2, smartshow

class ConvRBM(object):
    """
    Convolitional Restricted Bolzman Machine
    """

    def __init__(self, v_shape, n_hiddens, w_size=7, learning_rate=.05,
                 random_state=np.random, n_iter=10, verbose=False,
                 stride=(2, 2), n_gibbs=1, w_sigma=1, sparsity=.02):
        self.v_shape = v_shape
        self.w_size = w_size
        self.h_shape = (v_shape[0] - w_size + 1, v_shape[1] - w_size + 1)
        self.n_hiddens = n_hiddens
        self.n_iter = n_iter
        self.stride = stride
        self.n_gibbs = n_gibbs
        self.verbose = verbose
        self.lr = learning_rate
        self.rng = random_state
        self.stride = stride
        self.W = w_sigma * self.rng.normal(0, 1,
                                           (self.n_hiddens,) + (w_size, w_size))
        self.hiddens = self.rng.uniform(size=(self.n_hiddens,) + self.h_shape)
        # self.b = np.zeros((self.n_hiddens,) + self.h_shape)
        # self.c = np.zeros((self.v_shape[0] - 2*w_size + 2,
        #                    self.v_shape[1] - 2*w_size + 2))
        self.b = 0
        self.c = np.zeros((self.n_hiddens))
        self.dW = np.zeros(self.W.shape)
        self.db = self.b
        self.dc = np.zeros(self.c.shape)
        self.w_penalty = .05
        self.momentum = .9
        self.sparse_gain = 1
        self.sparsity = sparsity

    def fit(self, X):
        n_batches = X.shape[0]
        w, h = self.v_shape        
        for itr in xrange(self.n_iter):
            sum_err = 0
            # print('fit')
            for vi in xrange(n_batches):
                v = X[vi].reshape(self.v_shape)
                # print('fit2: ' + str(v.shape))
                dW, db, dc = self._gradients(v)
                self._apply_gradients(dW, db, dc)
                sum_err += self._batch_err(v)
            print('Iter %d: error = %d' % (itr, sum_err))
            dc_sparse = self.lr * self.sparse_gain * \
                        (self.sparsity - self.h_mean.mean(axis=2).mean(axis=1))
            self.c = self.c + dc_sparse

    def transform(self, v):
        I = np.zeros(self.h_mean.shape)
        for k in xrange(self.n_hiddens):
            I[k] = np.exp(conv2(v, self._ff(self.W[k]), 'valid') + self.c[k])
        pool_mean = 1 - (1. / (1 + self.pool(np.exp(I))))
        return pool_mean
        

    def pool(self, I):
        n_cols, n_rows = I.shape[1:]
        y_stride, x_stride = self.stride
        blocks = np.zeros(I.shape)
        for r in xrange(int(np.ceil(float(n_rows) / y_stride))):
            rows = range(r * y_stride, (r + 1) * y_stride)
            for c in xrange(int(np.ceil(float(n_cols) / x_stride))):
                cols = range(c * x_stride, (c + 1) * x_stride)
                block_val = I[:, rows, cols].sum()                
                block_val = np.swapaxes(np.swapaxes(block_val, 0, 1), 1, 2)
                blocks[:, rows, cols] = block_val
        return blocks
        
    def _gradients(self, v):
        v_mean, h_mean, h_mean0 = self._gibbs(v)
        # SIDE EFFECT: save means for future use (in _batch_err and
        # sparsity regularization)
        self.v_mean = v_mean
        self.h_mean = h_mean
        dW = np.zeros(self.W.shape)
        for k in xrange(self.n_hiddens):
            # print('_gradient: ' + str(v.shape))
            dW[k] = conv2(v, self._ff(h_mean[k]), 'valid') - \
                    conv2(v_mean, self._ff(h_mean[k]), 'valid')
        db = (v - v_mean).sum()
        dc = (h_mean0 - h_mean).sum(axis=2).sum(axis=1)  # TODO: check it! 
        return dW, db, dc

    def _apply_gradients(self, dW, db, dc):        
        self.W, self.dW = self._update_params(self.W, dW, self.dW, self.w_penalty)
        self.b, self.db = self._update_params(self.b, db, self.db, 0)
        self.c, self.dc = self._update_params(self.c, dc, self.dc, 0)
        
    def _update_params(self, params, grad, prev_grad, w_penalty):        
        grad = self.momentum * prev_grad + (1 - self.momentum) * grad
        params = params + self.lr * (grad - w_penalty * params);
        return params, grad

    def _batch_err(self, data):
        return ((data - self.v_mean) ** 2).sum()
        
    def _ff(self, mat):
        """
        Flip matrix both - from left to right and from up to down
        """
        return np.fliplr(np.flipud(mat))

        
    def _mean_hiddens(self, v):
        h = np.zeros((self.n_hiddens,) + self.h_shape)
        for k in xrange(self.n_hiddens):
            # print('_mean_hiddens (loop): %s' % (v.shape,))
            h[k] = np.exp(conv2(v, self._ff(self.W[k]), mode='valid') + self.c[k])
        h_mean = h / (1 + self.pool(h))
        return h_mean

    def _sample_hiddens(self, h_mean):
        return self._bernoulli(h_mean)

    def _mean_visible(self, h):
        v = np.zeros(self.v_shape)
        for k in xrange(self.n_hiddens):
            v += conv2(h[k], self.W[k], 'full')
        return logistic_sigmoid(v + self.b)
        
    def _sample_visible(self, h):        
        return self._bernoulli(_mean_visible)

    def _bernoulli(self, probs):
        res = np.zeros(probs.shape)
        res[self.rng.uniform(size=probs.shape) < probs] = 1
        return res
        
        
    def _gibbs(self, v0):
        h_full_shape = (self.n_hiddens,) + self.h_shape
        # print('gibbs: %s' % (v0.shape, ))
        h_mean0 = self._mean_hiddens(v0)
        h_mean = h_mean0
        for i in xrange(self.n_gibbs):            
            v_mean = self._mean_visible(self._bernoulli(h_mean))
            h_mean = self._mean_hiddens(v_mean)
        return v_mean, h_mean, h_mean0


def run_digits():
    import os
    from sklearn import datasets
    digits = datasets.load_digits()
    # custom_data_home = os.getcwd() + '/sk_data'
    # digits = datasets.fetch_mldata('MNIST original', data_home=custom_data_home)
    X = np.asarray(digits.data, 'float32')
    X /= 256
    crbm = ConvRBM((8, 8), 100, w_size=3, n_iter=3, verbose=True)
    crbm.fit(X)
    return crbm


def run_mnist():
    import os
    from sklearn import datasets
    custom_data_home = os.getcwd() + '/sk_data'
    digits = datasets.fetch_mldata('MNIST original', data_home=custom_data_home)
    X = np.asarray(digits.data, 'float32')
    X = X
    # images = [imresize(im.reshape(28, 28), (32, 32)) for im in X]
    # X = np.vstack([im.flatten() for im in images])
    X[X < 128] = 0
    X[X >= 128] = 1
    X /= 256.
    print('Building RBM')
    crbm = ConvRBM((28, 28), 40, w_size=11, n_iter=3, verbose=True)
    crbm.fit(X)
    return crbm




def test_configs():    
    from sklearn import datasets
    from datetime import datetime
    import sys
    import os    
    import logging
    log = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                            '%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    custom_data_home = os.getcwd() + '/sk_data'
    digits = datasets.fetch_mldata('MNIST original', data_home=custom_data_home)
    X = np.asarray(digits.data, 'float32')
    X = X
    # images = [imresize(im.reshape(28, 28), (32, 32)) for im in X]
    # X = np.vstack([im.flatten() for im in images])
    X[X < 128] = 0
    X[X >= 128] = 1
    X /= 256.
    models = []
    for w_sigma in [.1, .5, 1, 2, 5]:
        for sparsity in [.001, .01, .05, .1, .5]:
            log.info('Building RBM:\n  w_sigma=%s\n  sparsity=%s' %
                     (w_sigma,sparsity,))
            model = ConvRBM((28, 28), 40, w_size=11, n_iter=3, verbose=True,
                            w_sigma=w_sigma, sparsity=sparsity)
            model.fit(X)
            models.append({
                'model' : model,
                'w_sigma' : w_sigma,
                'sparsity' : sparsity,
            })
    log.info('Done')
    return models