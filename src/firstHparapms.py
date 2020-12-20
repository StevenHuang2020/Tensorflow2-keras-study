# python3
# steven keras test
import tensorflow as ts
from tensorflow import keras
import numpy as np
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import Dense, Flatten, Conv2D

from common import printModelWeights

print( ts.__version__)


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

def createModel(hparams):
    model = keras.Sequential()
    #model.add(Flatten())
    #model.add(Dense(units=1, input_shape=[1]))
    model.add(Dense(hparams[HP_NUM_UNITS], input_shape=[1]))
    #model.compile(optimizer='sgd', loss='mean_squared_error')
    model.add(Dense(1))
    
    model.compile(hparams[HP_OPTIMIZER], loss='mean_squared_error')
    model.summary()
    return model

def train(x,y):
    session_num = 0
    for num_units in HP_NUM_UNITS.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                #HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            model = createModel(hparams)
            model.fit(x, y, epochs=100)
            
            print('predict x=10, y=',model.predict([10.0]))
            #printModelWeights(model)
            
            session_num += 1
      
def main():
    # y = 2*x -1
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # model = createModel(hparams)
    # model.fit(x, y, epochs=100)
    
    # w, b = model.layers[0].get_weights()
    # print('weight %f , bias %f' % (w, b))
    # print(model.predict([10.0]))

    # printModelWeights(model)
    train(x,y)

if __name__ == "__main__":
    main()