#python3 steven tf 2.1.0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lys
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D


from common import getCsvDataset, printModelWeights
from plotCommon import plotTrainRes,plotLoss,plotLossDict

def funtion(X):
    return 1.58 * X**2 + 0.9

def preparDataSet(N=50, gamma=0.01):
    X = np.linspace(0, 3, N)
    X = np.around(X,decimals=2)
    noise = np.random.randn(X.shape[0]) * gamma
    y = funtion(X) + noise
    #y = np.around(y, decimals=2)
    return X, y

def predictTest(model, x_test, y_test):
    pred_test = model.predict(x_test)
    num = 5
    print('x_test=', x_test[:num])
    print('pred_test=', pred_test[:num].flatten())
    print('y_test=', y_test[:num].flatten())
    
def optimizerTf(lr=1e-3):
    #opt = optimizers.SGD(learning_rate=lr)
    opt = optimizers.RMSprop(learning_rate=lr)
    # opt = optimizers.Adam(learning_rate=lr)
    # opt = optimizers.Adadelta(learning_rate=lr)
    # opt = optimizers.Adagrad(learning_rate=lr)
    # opt = optimizers.Adamax(learning_rate=lr)
    # opt = optimizers.Nadam(learning_rate=lr)
    # opt = optimizers.Ftrl(learning_rate=lr)
    return opt

def create_model(opt, nFeatures=1):
    model = ks.models.Sequential()
    #lys = Dense(2, input_shape=(1,))
    #print(lys.get_weights())
    #print(lys.get_config())
    #print('nFeatures=', nFeatures)
    k_initializer = initializers.Ones()
    b_initializer = initializers.Ones()
    
    if 1:    # 1 layer
        model.add(Dense(5, input_shape=(nFeatures,), kernel_initializer=k_initializer,bias_initializer=b_initializer))
        model.add(lys.ReLU())
        model.add(Dense(5, kernel_initializer=k_initializer,bias_initializer=b_initializer))
        model.add(lys.ReLU())
        model.add(Dense(1, kernel_initializer=k_initializer,bias_initializer=b_initializer))
        
    else:    # multify layers
        model.add(Dense(10, input_shape=(nFeatures,)))
        model.add(Dense(8, input_shape=(10,)))
        model.add(Dense(2, input_shape=(8,)))
        model.add(Dense(1, input_shape=(2,)))
        
    model.compile(optimizer=opt,  loss='mean_squared_error') #metrics=['accuracy']
    #model.summary()
    return model

def trainModel(x_train,y_train,optimizer,epochs=50):
    nFeatures = x_train.shape[1]
    #print("*"*20,'nFeatures=',nFeatures)

    model = create_model(optimizer, nFeatures)
    #printModelWeights(model)
    history = model.fit(x=x_train, y=y_train, batch_size=10, shuffle=False, steps_per_epoch=True, verbose=0 ,epochs=epochs)
    #printModelWeights(model)
    losses = np.array(history.history['loss'])
    #print('losses=', losses)
    return losses, model
    
def main():
    #file = r'./db/fucDatasetReg_1F_NoLinear.csv'
    #file = r'./db/fucDatasetReg_2F.csv'
    file = r'./db/fucDatasetReg_3F_1000.csv'
    x_train, x_test, y_train, y_test = getCsvDataset(file)

    lr=1e-3
    EPOCHES = 200
    # optimizer = optimizerTf(lr=lr)
    # losses,_ = trainModel(x_train,y_train,optimizer,epochs=EPOCHES)
    # plotLoss(losses)

    opts = []
    # fast group
    opts.append((optimizers.SGD(learning_rate=lr), 'SGD'))
    opts.append((optimizers.RMSprop(learning_rate=lr), 'RMSprop'))
    opts.append((optimizers.Adam(learning_rate=lr), 'Adam'))
    opts.append((optimizers.Adamax(learning_rate=lr), 'Adamax'))
    opts.append((optimizers.Nadam(learning_rate=lr), 'Nadam'))
    # # slow group    
    opts.append((optimizers.Adadelta(learning_rate=lr), 'Adadelta'))
    opts.append((optimizers.Adagrad(learning_rate=lr), 'Adagrad'))
    opts.append((optimizers.Ftrl(learning_rate=lr), 'Ftrl'))
        
    lossesDict={}
    for opti,name in opts:
        losses,_ = trainModel(x_train,y_train,opti,epochs=EPOCHES)
        lossesDict[name] = losses
        #print(name, losses)
        
    plotLossDict(lossesDict)
        
if __name__=='__main__':
    main()
