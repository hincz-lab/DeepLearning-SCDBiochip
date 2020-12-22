from __future__ import division
import os 
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model, model_from_json


def standard_norm(img):
    height, width, channels = img.shape
    for channel in range(channels):
        img[:,:,channel] = (img[:,:,channel] - np.mean(img[:,:,channel]))/np.std(img[:,:,channel])
    return img

def call_model(path, filename):
    # load json and create model
    json_file = open(path + filename.replace('.h5','').replace('.json','') + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path + filename + '.h5')
    return loaded_model


def load_zoo(path_folder, model_filenames):
    ensemble = []
    for index, model_filename in enumerate(model_filenames):
        model = call_model(path_folder, model_filename)
        ensemble.append(model)
    return ensemble

def save_ensemble(ensemble_model, output_path):
    for kk, model in enumerate(ensemble_model):
        model_json = model.to_json()
        with open(output_path + f"/Kfold-model_{kk}" + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(output_path + f"/Kfold-model_{kk}" + ".h5")
    print("Saved model to disk")
    return 


def permute_DF(df):
    return df.sample(frac=1).reset_index(drop=True)


def append_storage(storage, cell_path, label):
    cell_size = len(os.listdir(cell_path))    
    for index, filename in enumerate(os.listdir(cell_path)):
        storage.append([cell_path + filename, label])
    return storage

# load list of channel names
def list_channels(path):
    return os.listdir(path)

def list_channels_df(path):
    df = pd.read_csv(path)
    channel_filenames = df.filename
    return channel_filenames


def create_TrainGen(df, column_names, img_height, img_width, bs = 32):

    datagen = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization=True,
              rotation_range = 90, horizontal_flip = True, vertical_flip = True)
    generator = datagen.flow_from_dataframe(dataframe = df, directory = None, 
            x_col = column_names[0], y_col = column_names[1:],
            has_ext = True, class_mode = 'raw', 
            target_size = (img_height, img_width), batch_size = bs)
    return datagen, generator


def create_TestGen(df, column_names, img_height, img_width, bs = 32):

    datagen = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization=True)
    generator = datagen.flow_from_dataframe(dataframe = df, directory = None, 
            x_col = column_names[0], y_col = column_names[1:], 
            has_ext = True, class_mode = 'raw', 
            target_size = (img_height, img_width), batch_size = bs)
    return datagen, generator


