from __future__ import division
import os 
import numpy as np
import pandas as pd
import PIL
from PIL import ImageOps
import matplotlib.pyplot as plt


import tensorflow.keras
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras import activations 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Lambda, concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.losses import binary_crossentropy
import tensorflow as tf

from tensorflow.keras import backend as K

# ============= Training Metrics/Losses ===============

def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy


def ce_jaccard_loss(y_true, y_pred):
    ce_loss = binary_crossentropy(y_true, y_pred)
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    jaccard_loss = -tf.math.log(intersection + K.epsilon()) / (union + K.epsilon())
    loss = ce_loss + jaccard_loss
    return loss

def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jaccard_coef(y_true,y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred)
    jac = (intersection + 1.) / (union - intersection + 1.)
    return K.mean(jac)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)




# ================= Phase 1 Models ====================

def Phase1_Net(img_size, num_classes):
    inputs = Input(shape=img_size + (3,))

    x = Conv2D(64,kernel_size = 3, strides = (1,1),
                            padding = "same")(inputs)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    
    previous_block_concatenate1 = x
    x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)

    x = Conv2D(128,kernel_size = 3, strides = (1,1),
                            padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)

    previous_block_concatenate2 = x

    concate_block_num = 3
    for filters in [256, 512, 512]:
        x = Conv2D(filters,3, strides = (1,1),
                            padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = Conv2D(filters,3, strides = 1,
                         padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)
        globals()['previous_block_concatenate%s' % concate_block_num] = x
        concate_block_num = concate_block_num + 1
        print(("No errors for filter size:" + str(filters)))



    x = Conv2D(512,3, strides = 1,
                            padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D(pool_size = (2,2),
                                  strides = (2,2))(x)

    x = Conv2D(512,3, strides = 1,
                            padding = "same")(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2DTranspose(256,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate5], axis =-1)

    x = Conv2DTranspose(256,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate4],axis=-1)

    x = Conv2DTranspose(128,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate3],axis=-1)
    
    x = Conv2DTranspose(64,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = concatenate([x, previous_block_concatenate2],axis=-1)


    x = Conv2DTranspose(32,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    x = Conv2DTranspose(64,2, strides = (2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)


    x = concatenate([x, previous_block_concatenate1],axis=-1)

    x = Conv2D(32,3, strides = (1,1),
                            padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = Conv2D(num_classes,3, strides = (1,1),
                            padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    outputs = Conv2D(num_classes,3, strides = (1,1),
                            activation = 'softmax',
                            padding = 'same',
                            name = 'sRBC_classes')(x)
    model = Model(inputs,outputs)

    return model

# ================ Train Phase 2 Model ================

# training the model for a specific amount of epochs 
def Phase1_train_network(model, X_train, y_train, 
                        X_test, y_test, epochs):
    
    train_history = model.fit(X_train, y_train, epochs=epochs, 
                              validation_data=(X_test, y_test),
                              shuffle = True, verbose = 2)
    return model, train_history

def Phase1_train_gen_network(model, x_gen, y_gen, epochs):
    train_history = model.fit(x_gen, epochs = epochs, validation_data = y_gen, verbose = 2, shuffle = True)
    return model, train_history

# ================= Phase 2 Models ====================
def Resnet50_model():
    return ResNet50(include_top = False, weights = "imagenet", input_shape = (224,224,3))

def Vgg16_model():
    return VGG16(include_top = False, weights = "imagenet", input_shape = (224,224,3))

def Xception_model():
    return Xception(include_top = False, weights = "imagenet", input_shape = (224,224,3))

def vanilla_model(target_size = (32,32)):

    input_layer = Input(shape = (224,224,3))
    x = Lambda(lambda image: tf.image.resize(image, target_size))(input_layer)
    x = Conv2D(32, 3, padding = "same")(x)
    x = MaxPooling2D(strides = 2)(x)
    x = Conv2D(64, 3, padding = "same")(x)
    x = MaxPooling2D(strides = 2)(x)
    model = tensorflow.keras.Model(inputs = input_layer, outputs=x, name="lower_head")
    return model



def classification_model(bottom_model, lr, num_class = 3):
    model = Sequential()
    model.add(bottom_model)
    model.add(GlobalAveragePooling2D(name = "FlobalGlobal_avg"))
    model.add(Dense(num_class, activation = "softmax", name = "sRBC_classes"))
    model.compile(optimizer = Adam(lr=lr), metrics = ["accuracy", "Precision", "Recall", "AUC"],
    loss = tensorflow.keras.losses.CategoricalCrossentropy())
    return model


# ================== Train Model ===========================
def train_model(model, train_gen, test_gen, epochs):

    STEP_SIZE_TRAIN = train_gen.n//train_gen.batch_size + 1
    STEP_SIZE_TEST = test_gen.n//test_gen.batch_size + 1

    train_history = model.fit_generator(generator = train_gen, steps_per_epoch = STEP_SIZE_TRAIN,
            validation_data = test_gen, validation_steps = STEP_SIZE_TEST, 
            epochs = epochs, shuffle = True, verbose = 1)
    return train_history, model







