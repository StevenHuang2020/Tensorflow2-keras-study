import tensorflow.keras as ks
from tensorflow.keras.layers import Dense, Flatten, Conv2D, RNN,LSTMCell
from tensorflow.keras.layers import BatchNormalization,LSTM
import tensorflow.keras.backend as K
import numpy as np 

from mnistCnn import prepareMnistData
from plotCommon import plotSubLossAndAcc

def createModel(input_shape, classes):
    model = ks.Sequential()
    
    if 1:
        lstm_layer = LSTM(64, input_shape=(None,28))
    else:
        lstm_layer = RNN(LSTMCell(64), input_shape=(None,28))
        
    model.add(lstm_layer)
    model.add(BatchNormalization())
    model.add(Dense(classes))

    model.summary()
        
    model.compile(optimizer='sgd', 
                  loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

def main():
    num_classes = 10
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData(0.1)
    print('x_train.shape = ',x_train.shape)
    print('input_shape.shape = ',input_shape)
    model = createModel(input_shape,num_classes)

    model.fit(x_train, y_train, epochs=5)
          #validation_data=(x_test, y_test),
          #batch_size=batch_size,
          
    
if __name__=="__main__":
    main()