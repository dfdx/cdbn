
import glob
import math

import cv2
import matplotlib.pylab as plt


def imshow(im):
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Image.fromarray(im).show()
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()
    
def smartshow(ims, subtitle='Images'):
    """
    Takes one image or list of images and tries to display them in a most
    convenient way.
    """
    if type(ims) != list:
        plt.figure()
        plt.imshow(ims, cmap=plt.cm.gray, interpolation='nearest')
        plt.show()
    else:
        n = len(ims)
        rows = math.floor(math.sqrt(n))
        cols = math.ceil(n / float(rows))
        plt.figure()
        for i, im in enumerate(ims):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(im, cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle(subtitle, fontsize=16)
        plt.show()

    
def list_images(path):
    return glob.glob(path + '/*.jpg') + \
        glob.glob(path + '/*.png') + \
        glob.glob(path + '/*.gif') + \
        glob.glob(path + '/*.pgm')
