#python3 steven tf 2.1.0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

class LinearModel:
    def __call__(self, x):
        return self.Weight * x + self.Bias

    def __init__(self):
        self.Weight = tf.Variable(1.0)
        self.Bias = tf.Variable(1.0)

def plotSubplot(x_train, y_train, w, b, loss):
    ax = plt.subplot(1, 2, 1)
    ax.title.set_text('training dataset')

    plt.scatter(x_train, y_train, label='input dataset')  # plot dataset

    x = np.linspace(0, 3, 10)  # plot result w & b
    y = w * x + b
    plt.plot(x, y, color='r')

    ax = plt.subplot(1, 2, 2)
    ax.title.set_text('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(len(loss)), loss)
    plt.show()


def plotSamplesAndPredict(x_train, y_train, w, b):
    plt.scatter(x_train, y_train, label='input dataset')  #plot  dataset

    x = np.linspace(0, 3, 10) #plot result w & b
    y = w*x + b
    plt.plot(x, y, color='r')
    plt.show()

def plotSamples(x, y):
    plt.scatter(x, y, label='input dataset')
    plt.show()

def plotEpochAndLoss(x, y):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, y)
    plt.show()

def preparDataSet(N=50):
    mSamples = N
    X = np.linspace(0, 3, mSamples)
    noise = np.random.randn(X.shape[0]) * 0.001
    y = 2 * X + 0.9 + noise
    return X, y

def lossFuc(y, pred):
    return tf.reduce_mean(tf.square(y - pred))#mse

def trainStep(linear_model, x, y, lr=0.001):
    with tf.GradientTape() as t:
        loss = lossFuc(y, linear_model(x))

    lr_weight, lr_bias = t.gradient(loss, [linear_model.Weight, linear_model.Bias])
    linear_model.Weight.assign_sub(lr * lr_weight)
    linear_model.Bias.assign_sub(lr * lr_bias)
    return loss

def train(model, epochs, X, y, lr=0.1):
    losses = []
    for epoch in range(epochs):
        loss = trainStep(model, X, y, lr=lr)

        w = model.Weight.numpy()
        b = model.Bias.numpy()
        #if epoch % 10 == 0:
        print(f"Epoch count {epoch}: Loss value: {loss.numpy()}, w:{w},b:{b}")
        losses.append(loss)
    return losses

def trainSGD(model, epochs, X, y, lr=0.1):
    losses = []
    for epoch in range(epochs):
        i = random.choice(range(len(X))) #np.random.choice(len(X),1)
        #print(i, X[i], y[i], range(len(X)))
        loss = trainStep(model, X[i], y[i], lr=lr)

        w = model.Weight.numpy()
        b = model.Bias.numpy()
        print(f"Epoch count {epoch}: Loss value: {loss.numpy()}, w:{w},b:{b}")
        losses.append(loss)
    return losses

def main():
    print("*"*100)
    #x, y = preparDataSet()
    X = np.array([-2,-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-5,-3.0,-1.0, 1.0, 3.0, 5.0, 7.0], dtype=float) #y = 2*x -1

    lr = 0.01
    epochs = 10
    model = LinearModel()

    #------GD---------
    #losses = train(model, epochs, X, y, lr=lr)

    #------SGD---------
    losses = trainSGD(model, epochs, X, y, lr=lr)

    w = model.Weight.numpy()
    b = model.Bias.numpy()
    #plotSamples(X,y)
    #plotSamplesAndPredict(X, y, w, b)
    #plotEpochAndLoss(epochs,losses)
    #plotSubplot(X, y, w, b, losses)

if __name__=='__main__':
    main()
