#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#Description: CNN visual #visual cnn kernel
#Date: 2020/12/21
#Author: Steven Huang, Auckland, NZ

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import backend as K
from mnistCnn import prepareMnistData
from cnnTf2Kernel import createModel,getKernelMatrix
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as Image
import cv2 

def showImgF(file):
    showImg(cv2.imread(file))

def showImg(img, saveFile=None):
    plt.imshow(img)
    if saveFile:
        plt.savefig(saveFile, format="svg")
    plt.show()

def printCNNModelLayers(model):
    for i,layer in enumerate(model.layers):
        if layer.name.startswith('conv'):
            try:
                filters, biases = layer.get_weights()
                print(i, layer.name, filters.shape, layer.trainable, layer.dtype)
            except:
                #print(i, layer.name, ' and no weights found!')
                pass
            
def getNeededLayers(model):
    printCNNModelLayers(model)
  
    layer_names = []
    layer_outputs = []
    layers = []
    #layersV = 4
    #layer_names = [layer.name for layer in model.layers[:layersV]] #all layers before flatten 
    #layer_outputs = [layer.output for layer in model.layers[:layersV]]

    for i,layer in enumerate(model.layers):
        if layer.name.startswith('conv2d'):
            layer_names.append(layer.name)
            layers.append(layer)
            layer_outputs.append(layer.output)
    return layer_names, layers, layer_outputs

def visualKernels(model): 
    """Visualizing trained kernels"""   
    layer_names, layers, _ = getNeededLayers(model)
    for name,layer in zip(layer_names, layers):
        filters, biases = layer.get_weights()
        print(name, filters.shape)
        
        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # plot first few filters
        n_filters, ix = filters.shape[-1], 1
        nums_row = 8
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            #print('f.shape:', f.shape)
            # plot each channel separately
            subs = f.shape[-1] #int(n_filters/nums_row)
            nums_col = int(subs/nums_row + 1)
            for j in range(subs):
                # specify subplot and turn of axis
                ax = plt.subplot(nums_row, nums_col, j+1)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(f[:, :, j], ) #cmap='gray'
     
        # show the figure
        plt.show()


def layerModelCreate(model, layer_name):
    #model.summary()
    #printCNNModelLayers(model)
    # Set up a model that returns the activation values for our target layer
    layer = model.get_layer(name=layer_name)
    return models.Model(inputs=model.inputs, outputs=layer.output)

def initialize_image(img_width, img_height, chn=3):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, chn))
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25

def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

def compute_loss(layerModel, input_image, filter_index):
    activation = layerModel(input_image) 
    #print('activation.shape:', activation.shape)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def gradient_ascent_step(layerModel, img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(layerModel, img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def visualize_filter(model, filter_index=0, layerName='conv2d', width=180, height=180, chn=1):
    #print('model.input=', model.input)
    #print('model.inputs=', model.inputs)

    layerModel = layerModelCreate(model,layer_name=layerName)
    iterations = 30
    learning_rate = 10.0
    img = initialize_image(img_width=width, img_height=height, chn=chn)
    
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(layerModel, img, filter_index, learning_rate)
        #print('loss=', loss)
    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    
    # file = r'./res/cnnTf2Kernel_0.png'
    # Image.save_img(file, img)
    # showImgF(file)    
    return loss, img

def show32FilterImg(model):
    name = 'conv2d_1' #'conv2d'
    
    all_imgs = []
    for filter_index in range(32):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(model, filter_index, layerName=name)
        all_imgs.append(img)

        #file = r'./res/' + name + '_stiched_filters_' + str(filter_index)+ '.png'
        #Image.save_img(file, img)
        
    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    img_width=180
    img_height=180
    
    margin = 1
    #n = 8
    row, col = 4,8
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = col * cropped_width + (col - 1) * margin
    height = row * cropped_height + (row - 1) * margin
    stitched_filters = np.zeros((width, height, 1))
    print('width, height:', width, height)
    
    # Fill the picture with our saved filters
    for i in range(col):
        for j in range(row):
            img = all_imgs[i * row + j]
            print(img.shape, i * col + j)
            stitched_filters[
                (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j : (cropped_height + margin) * j
                + cropped_height,
                :,
            ] = img
            
    file = r'./res/' + name + '_stiched_filters.png'
    Image.save_img(file, stitched_filters)
    showImgF(file)
    
def main():
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData(0.1) #input_shape 28*28*1
    model = createModel(input_shape, classes=10)

    file = r'./weights/cnnTf2Kernel_2.h5' #r'./weights/cnnTf2Kernel.h5'
    model.load_weights(file)
        
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc,'loss=',test_loss)
        
    #visualKernels(model)
    #visualize_filter(model)
    show32FilterImg(model)
    
if __name__=="__main__":
    main()
