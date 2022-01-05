#python3 steven tf 2.1.0
#01/04/2020
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,Conv2DTranspose
from tensorflow.keras.models import Sequential

from mnistCnn import prepareMnistData
from plotCommon import plotSubLossAndAcc

def createModel(input_shape, classes):
    model = Sequential()

    #output layer neurons shoule eqaul to the number of class
    #activation options: elu softmax selu softplus softsign relu tanh sigmoid hard_sigmoid exponential linear
    model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    #model.add(Conv2D(32, (3, 3), activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    if 1:#test
        model.add(Conv2DTranspose(filters=16, kernel_size=3, use_bias=False))

    model.add(Flatten())
    model.add(Dense(classes, activation='softmax')) #can not give input_shape,auto match


    lr = 0.01
    opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  #loss='mean_squared_error', #  #mean_absolute_error
                  #loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), #number class
                  loss=ks.losses.categorical_crossentropy, #one hot class
                metrics=['accuracy'])

    model.summary()
    return model

def extractSubwindowImages():
    n = 10
    # images is a 1 x 10 x 10 x 1 array that contains the numbers 1 through 100
    images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]

    print('images=\n',images)
    # We generate two outputs as follows:
    # 1. 3x3 patches with stride length 5
    # 2. Same as above, but the rate is increased to 2
    subs = tf.image.extract_patches(images=images,
                            sizes=[1, 5, 3, 1],
                            strides=[1, 5, 5, 1],
                            rates=[1, 1, 1, 1],
                            padding='VALID')
    print('subs=\n',subs.shape, subs)

    for i in subs:
        for j in i:
            print('j=\n',j.shape, j)

def main():
    #return extractSubwindowImages()

    num_classes = 10
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData(0.2) #input_shape 28*28*1
    model = createModel(input_shape,num_classes)
    history = model.fit(x=x_train, y=y_train, epochs=10,batch_size=200)
    #printModelWeights(model)

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc,'loss=',test_loss)
    loss = np.array(history.history['loss'])
    acc  = np.array(history.history['accuracy'])
    plotSubLossAndAcc(loss,acc)

if __name__=='__main__':
    main()
