
from __future__ import print_function
import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import helpers as hlp



def train_rbm(X, n_components=100, n_iter=10):
    X = X.astype(np.float64)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # scale to [0..1]
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = n_iter
    rbm.n_components = n_components
    rbm.fit(X)
    return rbm


############################


    
def run_autoencoder():
    imfiles = glob.glob('cropped/*.pgm')
    X = np.vstack([np.array(Image.open(fname).convert('L')).flatten()
                   for fname in imfiles])
    # digits = datasets.load_digits()
    # X = np.asarray(digits.data, 'float32')
    rbm = train_rbm(X, n_components=100, n_iter=5)
    images = [comp.reshape(192, 168) for comp in rbm.components_]
    images = images[:10]
    hlp.smartshow(images)
    