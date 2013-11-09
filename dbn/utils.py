
from operator import mul
import numpy as np
from scipy.signal import convolve2d

def random_uniform(shape, random_state=np.random, low=-0.5, high=0.5):
    total_size = 1
    for n in shape:
        total_size *= n
    return random_state.uniform(low, high, total_size).reshape(shape)

def convolve(im, k):
    return convolve2d(im, k, mode='valid')