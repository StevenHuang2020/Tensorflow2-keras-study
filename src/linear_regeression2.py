#python3 steven tf 2.1.0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lys
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from linear_regeression import plotSubplot


def preparDataSet(N=50):
    mSamples = N
    X = np.linspace(0, 3, mSamples)
    noise = np.random.randn(X.shape[0]) * 0.1
    y = 1.58 * X + 0.9 + noise
    #print(x)
    #print(y)
    return X, y

'''
class MyModel(Model):
      def __init__(self):
            super(MyModel, self).__init__()
            self.d1 = Dense(1, input_shape=(1,)) #Dense(128, activation='relu')

      def call(self, x):
            return self.d1(x)
'''

def create_model():
    model = ks.models.Sequential()
    lys = Dense(1, input_shape=(1,))
    #print(lys.get_weights())
    #print(lys.get_config())
    model.add(lys)
    #model.add(Dense(1))
    return model

def main():
    print("*"*100)
    x_train, y_train = preparDataSet()

    #model = create_model()
    #model = MyModel()
    model = ks.models.load_model('lLinear.model')

    opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    #opt = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    model.compile(optimizer='sgd',
                  loss='mean_squared_error' #  #mean_absolute_error
                  ) #metrics=['accuracy']
    history = model.fit(x=x_train, y=y_train, epochs=100)

    print(len(model.layers))
    weights = model.layers[0].get_weights()
    print(weights)
    w = weights[0].flatten()[0]
    b = weights[1].flatten()[0]
    #plotSamplesAndPredict(x_train, y_train, w, b)
    loss = history.history['loss']
    print(len(loss))
    epoch = np.arange(1,len(loss)+1)
    loss = np.array(loss)
    #print(epoch)
    #print(type(loss), loss)
    plotSubplot(x_train, y_train, w, b, loss)
    model.save('lLinear.model')

if __name__=='__main__':
    main()
