from __future__ import division
import os 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np 
import random
from tqdm import tqdm

from scipy import ndimage 
import pandas as pd


import tensorflow.keras
from tensorflow.keras.models import load_model, Model, Sequential

"""
     Implementing class activation maps for architectures with Global Average Pooling 2D before the finaly dense layer ...
"""


class CAM:

    def __init__(self, resize_width, resize_height):
        self.resize_height, self.resize_width = resize_height, resize_width

    # zero-center normalization
    def standard_norm(self, img):
        return ((img - np.mean(img))/np.std(img))

    # final layer should be (7,7,2048)
    def feature_model(self, model):
        feat_model = Sequential()
        feat_model = model.layers[0]
        return feat_model

    # final weight tensor before classification layer is 3*2048
    def weight_tensor(self, model):
        final_output = model.layers[-1] 
        return final_output.get_weights()[0]

    # output prediction class of the image of interest 
    def predict_class(self, model, X):
        prob_vec = model.predict(X)
        return np.argmax(prob_vec[0])

    # generate class activation maps (CAMs)
    def generate_CAM(self, model, img, label):
        norm_img = self.standard_norm(img)
        Fmap_model = self.feature_model(model)
        Wtensor = self.weight_tensor(model)
        feature_map = Fmap_model.predict(norm_img.reshape(1,224,224,3))
        CAM = feature_map.dot(Wtensor[:,label])[0,:,:]
        return cv.resize(CAM, (self.resize_width, self.resize_height), interpolation = cv.INTER_CUBIC)
    
    # generate prob vector
    def generate_probvec(self, model, img):
        X = self.standard_norm(img)
        prob_vec = model.predict(X.reshape(1,224,224,3))
        return prob_vec

"""
     Implementing image augmentations like rotation and flips ...
"""

class ManipulateCells:

    # class instance 
    def __init__(self, path, img_filename):
        self.img_name = img_filename

    # zero-center normalization
    def standard_norm(self, img):
        norm_img = (img - np.mean(img))/np.std(img)
        return norm_img, img

    # function courtesy of Alex Rodrigues from StackOverflow ...
    def rotate_image(self, img, angle):
        img_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv.getRotationMatrix2D(img_center, angle, 1.0)
        result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags = cv.INTER_LINEAR)
        return result
    
    # flip vertically 
    def flip_vertically(self, img):
        return cv.flip(img, 0)

    # flip horizontally 
    def flip_horizontally(self, img):
        return cv.flip(img, 1)

    # flip diagonally 
    def flip_diagonally(self, img):
        return cv.flip(img, -1)

    # augment the input image
    def augment_img(self, img):
        norm_img, orig_img = self.standard_norm(img)
        angle = random.randint(-180, 180)
        aug_img, orig_img = self.rotate_image(norm_img, angle), self.rotate_image(orig_img, angle)
        decision = random.randint(1,3)
        if decision == 1:
            aug_img, orig_img = self.flip_vertically(aug_img), self.flip_vertically(orig_img)
        elif decision == 2:
            aug_img, orig_img = self.flip_horizontally(aug_img), self.flip_horizontally(orig_img)
        elif decision == 3:
            aug_img, orig_img = self.flip_diagonally(aug_img), self.flip_diagonally(orig_img)
        return aug_img, orig_img









