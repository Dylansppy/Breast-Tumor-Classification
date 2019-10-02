import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#%matplotlib inline


import cv2
import os, time
import pandas as pd
import numpy as np
from keras.preprocessing import image
from preprocessing import list_pictures
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from multiprocessing import Pool

#Organizing the dataset
train_data_dir = os.path.join('..', 'data', 'train_processed')
test_data_dir = os.path.join('..', 'data', 'test_processed')
magnification_factors = ['40', '100', '200', '400']

model1 = Xception(weights='imagenet', include_top=False) #imports the Xception model and discards the last classification layer.
model2 = VGG16(weights='imagenet', include_top=False) #imports the VGG16 model and discards the last classification layer.
base_model = VGG19(weights='imagenet')
model3 = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output) #imports the VGG19 model and discards the last classification layer.
models = {'xception': (model1, xception_preprocess_input),
          'vgg16': (model2, vgg16_preprocess_input),
          'vgg19': (model3, vgg19_preprocess_input)
          }


# Define a function to extract features using pre-trained network
def feature_extraction(model_name, img_path, magnification_factor, input_shape, out_path):
    features = []
    labels = []
    for img_dir in list_pictures(img_path, magnification_factor):
        img = image.load_img(img_dir, target_size=input_shape)
        x = image.img_to_array(img) # (rows, columns. channels)
        # The network expects one or more images as input; that means the input array will need to be 4-dimensional:
        # samples, rows, columns, and channels.
        x = np.expand_dims(x, axis=0)
        x = models[model_name][1](x)
        feature = models[model_name][0].predict(x)
        feature_array = np.array(feature, dtype=float)
        features.append(feature_array)
        directory, file_name = os.path.split(img_dir)
        if file_name.split('_')[1] == 'B':
            label = 0
        elif file_name.split('_')[1] == 'M':
            label = 1
        labels.append(label)
    X = np.asarray(features)
    y = np.asarray(labels)
    np.save(os.path.join(out_path, magnification_factor, 'X.npy'), X)
    np.save(os.path.join(out_path, magnification_factor, 'y.npy'), y)

# Deply feature extraction
def feature_extraction_deploy(model_name, input_shape, train_output_dir, test_output_dir,
                              train_input_dir=train_data_dir, test_input_dir=test_data_dir):
    for i in range(4):
        feature_extraction(model_name, train_input_dir, magnification_factors[i], input_shape, train_output_dir)
        feature_extraction(model_name, test_input_dir, magnification_factors[i], input_shape, test_output_dir)

# Extract features with Xception, the default input size for this model is 299x299.
img_rows, img_cols= 299, 299
input_shape = (img_rows, img_cols)
out_path_xception_train = os.path.join('..', 'data', 'features_xception', 'Train')
out_path_xception_test = os.path.join('..', 'data', 'features_xception', 'Test')
feature_extraction_deploy('xception', input_shape, out_path_xception_train, out_path_xception_test)

# Extract features with VGG16, the default input size for this model is 224x224.
img_rows, img_cols= 224, 224
input_shape = (img_rows, img_cols)
out_path_vgg16_train = os.path.join('..', 'data', 'features_vgg16', 'Train')
out_path_vgg16_test = os.path.join('..', 'data', 'features_vgg16', 'Test')
feature_extraction_deploy('vgg16', input_shape, out_path_vgg16_train, out_path_vgg16_test)


# Extract features from an arbitrary intermediate layer with VGG19, the default input size for this model is 224x224
img_rows, img_cols= 224, 224
input_shape = (img_rows, img_cols)
out_path_vgg19_train = os.path.join('..', 'data', 'features_vgg19', 'Train')
out_path_vgg19_test = os.path.join('..', 'data', 'features_vgg19', 'Test')
feature_extraction_deploy('vgg19', input_shape, out_path_vgg19_train, out_path_vgg19_test)


