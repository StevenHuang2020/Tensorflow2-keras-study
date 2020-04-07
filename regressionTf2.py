#python3 steven tf 2.1.0
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lys
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from common import getCsvDataset, printModelWeights
from plotCommon import plotTrainRes,plotLoss

def funtion(X):
    return 1.58 * X**2 + 0.9

def preparDataSet(N=50):
    mSamples = N
    X = np.linspace(0, 3, mSamples)
    X = np.around(X,decimals=2)
    noise = np.random.randn(X.shape[0]) * 0.1
    y = funtion(X) + noise
    y = np.around(y, decimals=2)
    return X, y

def prepareTest():
    mSamples = 10
    X = np.linspace(0, 3, mSamples)
    X = np.around(X, decimals=2)
    #y = funtion(X)
    return X

def create_model(nFeatures=1):
    model = ks.models.Sequential()
    #lys = Dense(2, input_shape=(1,))
    #print(lys.get_weights())
    #print(lys.get_config())
    
    if 0:    # 1 layer
        model.add(Dense(1, input_shape=(nFeatures,)))
    else:    # multify layers
        model.add(Dense(10, input_shape=(nFeatures,)))
        model.add(Dense(8, input_shape=(10,)))
        model.add(Dense(2, input_shape=(8,)))
        model.add(Dense(1, input_shape=(2,)))
    
    lr = 0.001
    opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  loss='mean_squared_error' #  #mean_absolute_error
                  ) #metrics=['accuracy']
    
    model.summary()
    return model

def main():
    if 0:
        x_train, y_train = preparDataSet()
        x_test = prepareTest()
        print(x_train.shape)
        print(y_train.shape)
    else:
        #file = r'./db/fucDatasetReg_2F.csv'
        file = r'./db/fucDatasetReg_3F.csv'
        x_train, x_test, y_train, y_test = getCsvDataset(file)

    nFeatures = x_train.shape[1]
    print("*"*100,'nFeatures=',nFeatures)
    model = create_model(nFeatures)
    history = model.fit(x=x_train, y=y_train, epochs=50)
    #printModelWeights(model)
    
    loss = history.history['loss']
    loss = np.array(loss)
 
    pred_test = model.predict(x_test)
    
    num = 5
    print('x_test=', x_test[:num])
    print('pred_test=', pred_test[:num].flatten())
    print('y_test=', y_test[:num].flatten())
    #plotTrainRes(x_train, y_train,x_test, pred_test, loss)
    plotLoss(loss)

if __name__=='__main__':
    main()


