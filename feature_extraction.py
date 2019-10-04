import os
import time
import numpy as np
from multiprocessing import Pool
from keras.preprocessing import image
import cv2
from keras.models import Model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from preprocessing import list_pictures


#Organizing the dataset
image_dir = os.path.join('..', 'data', 'image_processed')
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
    patients = []
    for img_dir in list_pictures(img_path, magnification_factor, ext='.npy'):
        x = np.load(img_dir) # 460*700*7
        x = cv2.resize(x, dsize=input_shape, interpolation=cv2.INTER_CUBIC) #(input_size, 7)
        # img = image.load_img(img_dir, target_size=input_shape)
        # x = image.img_to_array(img)  # (rows, columns. channels)
        # The network expects one or more images as input; that means the input array will need to be 4-dimensional:
        # samples, rows, columns, and channels.  channel number of 3 is required for pre-trained model
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
        patients.append(file_name.split('-')[2])
    X = np.asarray(features)
    y = np.asarray(labels)
    p = np.asarray(patients)
    feature_path = os.path.join(out_path, magnification_factor)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    np.save(os.path.join(feature_path, 'X.npy'), X)
    np.save(os.path.join(feature_path, 'y.npy'), y)
    np.save(os.path.join(feature_path, 'p.npy'), p)


# Deploy feature extraction
def feature_extraction_deploy(model_name, input_shape, output_dir, input_dir=image_dir):
    multi1 = time.time()
    print('Parent process {0}'.format(os.getpid()))
    print('Waiting for all processes done...')
    p = Pool(4)
    for i in range(4):
        p.apply_async(feature_extraction,
                      args=(model_name,
                            input_dir,
                            magnification_factors[i],
                            input_shape,
                            output_dir
                            )
                      )
    p.close()
    p.join()
    print('All processes done.')
    multi2 = time.time()
    print('Time comsumed:', multi2 - multi1)

if __name__ == "__main__":
    # Extract features with Xception, the default input size for this model is 299x299.
    input_shape = (299, 299)
    out_path_xception = os.path.join('..', 'data', 'features_xception')
    feature_extraction_deploy('xception', input_shape, out_path_xception)

    # Extract features with VGG16, the default input size for this model is 224x224.
    input_shape = (224, 224)
    out_path_vgg16 = os.path.join('..', 'data', 'features_vgg16')
    feature_extraction_deploy('vgg16', input_shape, out_path_vgg16)

    # Extract features from an arbitrary intermediate layer with VGG19, the default input size for this model is 224x224
    input_shape = (224, 224)
    out_path_vgg19 = os.path.join('..', 'data', 'features_vgg19')
    feature_extraction_deploy('vgg19', input_shape, out_path_vgg19)



