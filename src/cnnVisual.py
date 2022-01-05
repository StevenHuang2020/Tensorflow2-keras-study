#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#Description: CNN visual #1.visual feature map 2.visual cnn kernel 3.visual heat map
#Date: 2020/12/20
#Author: Steven Huang, Auckland, NZ

#Referece:https://cs231n.github.io/understanding-cnn/
#Deep Learning with Python.pdf - 160
#https://github.com/conan7882/CNN-Visualization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras import models
from mnistCnn import prepareMnistData
from cnnTf2Kernel import createModel
import matplotlib.pyplot as plt

''' matplot colormap optional values
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', \
'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', \
'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',\
'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', \
'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', \
'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', \
'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', \
'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', \
'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', \
'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', \
'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', \
'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', \
'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', \
'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', \
'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', \
'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', \
'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', \
'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', \
'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', \
'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', \
'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', \
'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', \
'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
'''

def plotOneActivation(activations):
    op = activations[0] #(1, 26, 26, 32)
    plt.matshow(op[0, :, :, 0], cmap='viridis')#viridis
    plt.show()

    #op = activations[1] #(1, 13, 13, 32)
    #plt.matshow(op[0, :, :, 0], cmap='viridis')#viridis
    #plt.show()

def plotAllActivation(activations, names):
    images_per_row = 8
    for layer_name, layer_activation in zip(names, activations):
        n_features = layer_activation.shape[-1]
        h = layer_activation.shape[1]
        w = layer_activation.shape[2]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((h * n_cols, images_per_row * w))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * h : (col + 1) * h,
                    row * w : (row + 1) * w] = channel_image

        scaleH,scaleW = 1./h, 1./w
        plt.figure(figsize=(scaleW * display_grid.shape[1],  scaleH * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

def getNeededLayers(model):
    for i,layer in enumerate(model.layers):
        if layer.name.startswith('conv2d'):
            print(i, layer.name, layer.trainable, layer.dtype)

    layer_names = []
    layer_outputs = []
    #layersV = 4
    #layer_names = [layer.name for layer in model.layers[:layersV]] #all layers before flatten
    #layer_outputs = [layer.output for layer in model.layers[:layersV]]

    for i,layer in enumerate(model.layers):
        if layer.name.startswith('conv2d'):
            layer_names.append(layer.name)
            layer_outputs.append(layer.output)
    return layer_names, layer_outputs

def visualModel(model, img):
    """Visualizing intermediate activations"""
    img = img.reshape(-1, 28, 28, 1)

    layer_names, layer_outputs = getNeededLayers(model)

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)
    print('layer_names:', layer_names)
    print('activations:', type(activations), len(activations))
    for i,op in enumerate(activations):
        print(i, type(op), op.shape)

    #plotOneActivation(activations)
    plotAllActivation(activations, layer_names)

def main():
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData(0.1) #input_shape 28*28*1
    model = createModel(input_shape, classes=10)

    file = r'./weights/cnnTf2Kernel_2.h5' #r'./weights/cnnTf2Kernel.h5'
    model.load_weights(file)

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc,'loss=',test_loss)

    testImg = x_test[6]
    testLabel = y_test[6]
    # print('testImg:', testImg.shape)
    # print('testLabel:', testLabel)
    # plt.imshow(testImg)
    # plt.show()

    visualModel(model, testImg)

if __name__=="__main__":
    main()
