
import glob
import os

import numpy as np
import matplotlib.pylab as plt
import cv2
from PIL import Image

import helpers as hlp


def detect(im, cascade=None, cascade_xml='haarcascade_frontalface_alt2.xml'):
    if len(im.shape) != 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if not cascade:
        cascade = cv2.CascadeClassifier(cascade_xml)
    im = cv2.equalizeHist(im)
    faces = cascade.detectMultiScale(im)
    return faces
    
    

############################
    
def run():
    for fname in hlp.list_images('cropped'):
        im = cv2.imread(fname)
        if im == None: continue
        faces = detect(im)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y),
                          (x + w, y + h), (255, 0, 0), 3)
        imshow(im)