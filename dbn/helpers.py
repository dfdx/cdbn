
import glob
import math
import numpy as np

import cv2
import matplotlib.pylab as plt


def imshow(im):
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Image.fromarray(im).show()
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()
    
