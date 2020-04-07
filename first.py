# python3
# steven keras test
from tensorflow import keras
import numpy as np

from common import printModelWeights

def train(x, y, epoch):
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=epoch)
    return model

def main():
    # y = 2*x -1
    x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model = train(x, y, 500)

    w, b = model.layers[0].get_weights()
    print('weight %f , bias %f' % (w, b))
    print(model.predict([10.0]))

    printModelWeights(model)
    pass

if __name__ == "__main__":
    main()