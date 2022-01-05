#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#Description: CNN visual #visual class activation visualization heat map
#Date: 2020/12/22
#Author: Steven Huang, Auckland, NZ
#Reference: https://keras.io/examples/vision/grad_cam/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cnnVisualHeatmap import make_gradcam_heatmap,getCombineImg

model_builder = keras.applications.xception.Xception

preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def getHeatMap(file, model):
    img_array = preprocess_input(get_img_array(img_path = file, size=(299, 299)))
    preds = model.predict(img_array)
    print("Predicted:", decode_predictions(preds, top=1)[0])

    last_conv_layer_name = "block14_sepconv2_act"
    classifier_layer_names = [
        "avg_pool",
        "predictions",
    ]
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
    return heatmap

def main():
    file = r'./res/african_elephant.jpg'
     # Make model
    model = model_builder(weights="imagenet")
    model.summary()

    heatmap = getHeatMap(file, model)
    # # Display heatmap
    # plt.matshow(heatmap)
    # plt.show()

    img = keras.preprocessing.image.load_img(file)
    img = keras.preprocessing.image.img_to_array(img)
    getCombineImg(img, heatmap, r'./res/african_elephantHeat.jpg')

if __name__=="__main__":
    main()
