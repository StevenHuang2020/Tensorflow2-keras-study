#python3 steven tf 2.1.0
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lys
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from common import getCsvDataset, printModelWeights
from plotCommon import plotTrainRes,plotLoss,plotLossAndAcc,plotSubLossAndAcc


def create_model(nFeatures=1):
    model = ks.models.Sequential()

    #output layer neurons shoule eqaul to the number of class
    #activation options: elu softmax selu softplus softsign relu tanh sigmoid hard_sigmoid exponential linear 
    if 0:    # 1 layer
        model.add(Dense(3, input_shape=(nFeatures,),activation='softmax')) 
    else:    # multify layers
        model.add(Dense(10, input_shape=(nFeatures,)))
        model.add(Dense(10))
        model.add(Dense(10))
        model.add(Dense(8, activation='softmax')) #can not give input_shape,auto match
        model.add(Dense(3))
     
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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    model.summary()
    return model

def main():
    #file = r'./db/fucDatasetClf_2F.csv'
    file = r'./db/fucDatasetClf_2F_MClass.csv'
    
    x_train, x_test, y_train, y_test = getCsvDataset(file)

    nFeatures = x_train.shape[1]
    print("*"*100,'nFeatures=',nFeatures)
    
    model = create_model(nFeatures)
    history = model.fit(x=x_train, y=y_train, epochs=50)
    #printModelWeights(model)
    
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc,'loss=',test_loss)

    
    num = 5
    predProb_test = model.predict(x_test)
    pred_test = predProb_test.argmax(axis=-1)

    print('x_test=', x_test[:num])
    print('pred_test=', pred_test[:num])
    print('y_test=', y_test[:num].flatten())
    #plotTrainRes(x_train, y_train,x_test, pred_test, loss)
    loss = np.array(history.history['loss'])
    acc  = np.array(history.history['accuracy'])
    #print(acc)
    #plotLoss(loss)
    plotSubLossAndAcc(loss,acc)

if __name__=='__main__':
    main()


