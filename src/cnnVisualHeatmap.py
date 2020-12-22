#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#Description: CNN visual #visual heat map
#Date: 2020/12/22
#Author: Steven Huang, Auckland, NZ
#Reference: https://keras.io/examples/vision/grad_cam/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow import keras
from mnistCnn import prepareMnistData
from cnnTf2Kernel import createModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def getCombineImg(img, heatmap, saveFile=None):
      # We rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    if saveFile:
        superimposed_img.save(saveFile)

def getHeatMap(model, img): 
    """Visualizing class activation map(CAM)"""
    img = img.reshape(-1, 28, 28, 1)
    preds = model.predict(img)
    label = np.argmax(preds[0])
    print('label,preds, preds.shape:', label, preds, preds.shape)

    last_conv_layer_name = "conv2d_1"
    classifier_layer_names = [
        "max_pooling2d_1",
        "flatten",
        "dense"
    ]
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img, model, last_conv_layer_name, classifier_layer_names)
    return heatmap


def main():
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData(0.1) #input_shape 28*28*1
    model = createModel(input_shape, classes=10)

    file = r'./weights/cnnTf2Kernel_2.h5' #r'./weights/cnnTf2Kernel.h5'
    model.load_weights(file)
        
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc,'loss=',test_loss)
    
    testImg = x_test[6]
    testLabel = y_test[6]
    print('testLabel:', testLabel)
    
    heatmap = getHeatMap(model, testImg)
    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    
    getCombineImg(testImg,heatmap, r'./res/7.png')
    
if __name__=="__main__":
    main()
