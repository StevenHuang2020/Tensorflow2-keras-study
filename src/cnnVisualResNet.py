#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#Description: ResNet Visual
#Date: 2020/12/21
#Author: Steven Huang, Auckland, NZ
#Reference: https://keras.io/examples/vision/visualizing_what_convnets_learn/

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
from cnnVisualKernel import printCNNModelLayers
from cnnVisualKernel import compute_loss, gradient_ascent_step,initialize_image
from cnnVisualKernel import deprocess_image,showImg,visualize_filter,layerModelCreate
import cv2
import tensorflow.keras.preprocessing.image as Image

def createModel():
     # Build a ResNet50V2 model loaded with pre-trained ImageNet weights
    model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
    model.summary()
    #printCNNModelLayers(model)
    return model

def showOneFilterImg(model):
    name = 'conv5_block3_out' #'conv5_block2_out' #'conv5_block1_out' 
    #'conv4_block6_out' #'conv4_block4_out' #'conv3_block4_out'
    loss, img = visualize_filter(model, 0, layerName=name, width=512, height=512, chn=3)
    file = r'./res/' + name + '_0.png'
    Image.save_img(file, img)
    showImg(img)    
    
def show64FilterImg(model):
    # Compute image inputs that maximize per-filter activations
    # for the first 64 filters of our target layer
    name = 'conv3_block4_out'
    
    all_imgs = []
    for filter_index in range(64):
        print("Processing filter %d" % (filter_index,))
        loss, img = visualize_filter(model, filter_index, layerName=name,chn=3)
        all_imgs.append(img)

        #file = r'./res/' + name + '_stiched_filters_' + str(filter_index)+ '.png'
        #Image.save_img(file, img)
        
    # Build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    img_width=180
    img_height=180
    
    margin = 5
    n = 8
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = n * cropped_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = all_imgs[i * n + j]
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
    model = createModel()
    showOneFilterImg(model)
    #show64FilterImg(model)
    
if __name__=="__main__":
    main()
