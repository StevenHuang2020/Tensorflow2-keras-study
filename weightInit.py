#python3
#steven keras test
import tensorflow.keras as ks
import numpy as np 
from tensorflow.keras.callbacks import LambdaCallback,ModelCheckpoint
from tensorflow.keras import initializers
import matplotlib.pyplot as plt

from plotCommon import plotTrainRes,plotLossAndAcc,plotSubLossAndAcc
from plotCommon import plotLoss, plotLosses, plotLossAx,plotLosseList,plotLosseListTu

def createModel(initWeight=0):
    model = ks.Sequential()
    #model.add(ks.layers.Dense(units=1, input_shape=[1])
    #model.add(ks.layers.Dense(units=1, input_shape=[1],kernel_initializer='zeros')) #initializers
    #model.add(ks.layers.Dense(units=1, input_shape=[1], kernel_initializer='ones')) #bias_initializer
    model.add(ks.layers.Dense(units=1, input_shape=[1], kernel_initializer=initializers.constant(initWeight)))

    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.summary()
    return model
    
def main():
    #y = 2*x -0 #1
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    #y = np.array([-3.0,-1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    y = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0], dtype=float)

    initWeights=[0,1.2,2,2.5]
    lossLabels=[]
    for i,Weight in enumerate(initWeights):
        model = createModel(Weight)
        print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
        #checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=False)

        #model.fit(x,y, epochs=10,callbacks = [print_weights,checkpointer])
        history = model.fit(x,y, epochs=10,callbacks = [print_weights])
        #history = model.fit(x, y, epochs=10)

        w, b = model.layers[0].get_weights()
        print('weight %f , bias %f' % (w, b))
        #print(model.predict([10.0]))

        loss = history.history['loss']
        #print(loss)
        #plotLoss(loss)
        lossLabels.append((loss,'init weight='+str(Weight)))

    plotLosseListTu(lossLabels)

if __name__=="__main__":
    main()
