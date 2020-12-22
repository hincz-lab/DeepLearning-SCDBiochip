from __future__ import division
import os
import cv2 as cv
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
import pandas as pd
from numba import jit
import time
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import tensorflow.keras
from tensorflow.keras.models import load_model
import tensorflow as tf


class CountAdheredBloodCells:

    alpha,beta = 35, 100 # class variables (amount of channel paritions
    
    # instance class
    def __init__(self, path, channel_filename):
        self.channel_image = cv.imread(path + channel_filename)


    # crop individual tile images
    def process_tiles(self, kk=0):
        import cv2 as cv
        X = np.zeros((self.alpha*self.beta, 128, 128, 3))
        for ii in range(self.alpha):
            for jj in range(self.beta):
                y_slider, x_slider = ii*150, jj*150
                image = self.channel_image[0+y_slider:150+y_slider, 0+x_slider:150+x_slider,:]
                X[kk,:,:,:] = cv.resize(image, (128,128), interpolation = cv.INTER_CUBIC).reshape(128,128,3)
                kk+=1
        print("Total number of extracted tiles: ", len(X[:,0,0,0]))
        return X

    # zero-center normalization for both Phase I and II 
    def standard_norm(self, tile, phase):
        norm_tile = (tile - np.mean(tile))/(np.std(tile))
        if phase == 1:
            norm_tile = norm_tile.reshape(1,128,128,3)
        if phase == 2:
            norm_tile = norm_tile.reshape(1,224,224,3)
        
        return norm_tile
    
    # output segmentation masks (one hot encoded probability tensors)
    @tf.function
    def Phase1_prediction(self, ensemble, tile):
        sample, height, width, depth = tile.shape
        empty_mask = np.zeros((sample, height, width, depth))
        for kfold, model in enumerate(ensemble):
            empty_mask += model(tile)
        empty_mask *= 1/len(ensemble)
        return empty_mask
        
    # convert the predicted one hot encoded masks to label masks    
    def predict_masks(self, X, ensemble, phase = 1):
        y_preds = []
        for sample in range(self.alpha*self.beta):
            tile = np.reshape(X[sample,:,:,:].astype('float32'), (1,128,128,3))
            norm_tile = self.standard_norm(tile, phase)
            mask = self.Phase1_prediction(ensemble, norm_tile)
            mask = np.argmax(mask, axis = 3)
            mask = np.reshape(mask, (128,128,1))[:,:,0]
            y_preds.append(mask)
        return y_preds
    
    # concatenate back the individual segmentation masks into a whole channel mask
    def preprocess_channel_mask(self, X, ensemble, kk=0):
        channel_mask = np.zeros((self.alpha*150, self.beta*150))
        y_preds = self.predict_masks(X, ensemble)
        self.masks = y_preds
        for ii in range(self.alpha):
            for jj in range(self.beta):
                y_slider, x_slider = ii*150, jj*150
                pred_mask = y_preds[kk].astype('uint8')
                pred_mask = cv.resize(pred_mask, (150,150), interpolation = cv.INTER_CUBIC)
                channel_mask[0+y_slider:150+y_slider, 0+x_slider:150+x_slider] = pred_mask
                kk+=1
        self.mask = channel_mask
        return channel_mask.astype('uint8')
    
#    Class methods for preparing the input data in Phase 2 is found below ...
    
    # generate binary masks with the foreground as the adhered BC label
    def binarize_cells(self, channel_mask):
        mask_cells = channel_mask == 1
        mask_cells = mask_cells*1
        return mask_cells
    
    # cleaning up images after generating centroids on top of the image (might be redundant ...)
    def remove_centroid_indent(self, crop_cell):
        crop_cell[:,:,0] = crop_cell[:,:,1]
        return crop_cell
    
    @staticmethod          
    @jit(nopython=True)
    def compute_labelsizes(Nlabels,Zlabeled):
        return [(Zlabeled == label).sum() for label in range(Nlabels + 1)]

    @staticmethod
    @jit(nopython=True)
    def clean_masks(label_size, Zlabeled, mask_cells, threshold = 60):

        bin_mask = np.zeros(mask_cells.shape)
        for label in range(len(label_size)):
            if label_size[label] > threshold:
                bin_mask += (Zlabeled == label)*1.0
        return bin_mask

    def ext_IndividualCells(self, channel_mask, padding = 20, crop_size = 32, count = 0):        
        img_borders = cv.copyMakeBorder(self.channel_image.copy(), padding, padding, padding, padding, cv.BORDER_CONSTANT)
        binary_mask = (channel_mask == 1)*1
        blobLabels = measure.label(binary_mask)
        labelProperties = measure.regionprops(blobLabels)
        centroids = [prop.centroid for prop in labelProperties if prop.area > 60]
        img_container, tot_times = [], []
        for centroid in centroids:
            centroid_x,centroid_y = int(round(centroid[0],0)), int(round(centroid[1],0))
            bound_left, bound_right, bound_bottom, bound_top = centroid_x - int(crop_size/2), centroid_x + int(crop_size/2), centroid_y - int(crop_size/2), centroid_y + int(crop_size/2)
            # corret for the padding
            bound_left += padding
            bound_right += padding
            bound_bottom += padding
            bound_top += padding
            if bound_left<0 and bound_right>0: 
                bound_right -= bound_left
                bound_left -= bound_left
            elif bound_left>0 and bound_right<0:
                bound_left -= bound_right
                bound_right -= bound_right
            if bound_bottom<0 and bound_top>0:
                bound_top -= bound_bottom
                bound_bottom -= bound_bottom
            elif bound_bottom>0 and bound_top<0:
                bound_bottom -= bound_top
                bound_top -= bound_top
            cell = img_borders[bound_left:bound_right, bound_bottom:bound_top,:] 
            img_container.append(cell)     
        return img_container

#    Class methods for implementing Phase 2 is found below ...

    @tf.function
    def predict_class(self, ensemble, tile):
        y_pred = np.zeros((3,))
        for kfold, model in enumerate(ensemble):
            y_pred += model(tile)
        y_pred *= 1/len(ensemble)
        return y_pred


    # predict one hot encoded probability vectors in Phase II
    def Phase2_prediction(self, ensemble, tiles):
        samples, height, width, depth = tiles.shape
        y_preds = np.zeros((samples, 3))
        for sample in range(samples):
            y_preds[sample,:] = self.predict_class(ensemble, tiles[sample,:,:].reshape(1,height,width,depth))


        #    for kfold, model in enumerate(ensemble): 
       #         y_preds[sample,:] += model.predict(tiles[sample,:,:,:].reshape(1,height,width,depth)).reshape(3,)
        #    y_preds[sample,:] *= 1/len(ensemble)
       
        return y_preds
    
    # count the prediction based on the max probability within an individual vector
    def count_classes(self, predictions, rbc_thres, wbc_thres, other_thres):
        #predictions = np.argmax(predictions, axis=1)
        sRBC = 0
        WBC = 0
        other = 0
        sRBC_container, WBC_container, Other_container = [],[],[]
        for sample in range(len(predictions)):
            if predictions[sample][0] >= rbc_thres[0]:
                sRBC+=1
                sRBC_container.append(sample)
            if predictions[sample][1] >= wbc_thres[0]:
                WBC+=1
                WBC_container.append(sample)
            if predictions[sample][2] >= other_thres[0]:
                other+=1
                Other_container.append(sample)
    
        return sRBC,WBC,other, sRBC_container, WBC_container, Other_container    

    # function that encompasses all the neccesary functions to complete Phase II 
    def count_predictions(self, ensemble, img_container, rbc_thres, wbc_thres, other_thres, phase=2):
        X_Phase2 = np.zeros((len(img_container), 224, 224, 3))
        norm_X_Phase2 = np.zeros((len(img_container), 224, 224, 3))
        for sample, image in enumerate(img_container):
            X_Phase2[sample,:,:,:] = cv.resize(image.astype('float32'), (224,224), interpolation=cv.INTER_CUBIC)*1.0/255.0
            norm_X_Phase2[sample,:,:,:] = self.standard_norm(X_Phase2[sample,:,:,:], phase)
        y_preds_Phase2 = self.Phase2_prediction(ensemble, norm_X_Phase2)
        sRBC,WBC,Other, sRBC_container, WBC_container, Other_container = self.count_classes(y_preds_Phase2, rbc_thres, wbc_thres, other_thres)
        return sRBC, WBC, Other, sRBC_container, WBC_container, Other_container

#    Call the pipeline for cell counting ...

    # main function for cell counting
    def call_pipeline(self, Phase1_ensemble, Phase2_ensemble,rbc_thres, wbc_thres, other_thres):
        # Prepare the Phase I data ...
        print('Prepare the Phase I data ...')
        X = self.process_tiles()
        samples, height, width, depth = X.shape
        norm_X = np.zeros((samples, height, width, depth))
        for sample in range(samples):
            norm_X[sample, :, :, :] = self.standard_norm(X[sample, :, :, :], 1)
        print('Complete ...')
        # Implement Phase I ...
        print('Implementing Phase I ...')
        channel_mask = self.preprocess_channel_mask(norm_X, Phase1_ensemble, kk=0)
        print('Complete ...')
        # Preprare the Phase II data ...
        print('Prepare Phase II data ...')
        img_container = self.ext_IndividualCells(channel_mask)
        print('Complete ...')
        # Implement Phase II ...
        print('Implementing Phase II ...')
        sRBC, WBC, Other, sRBC_container, WBC_container, Other_container = self.count_predictions(Phase2_ensemble, img_container, rbc_thres, wbc_thres, other_thres)
        print('Complete ...\n')
        return sRBC, WBC, Other, img_container, sRBC_container, WBC_container, Other_container
    

        return
       
