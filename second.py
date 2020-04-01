import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def prepareDataSet(N=200):
    x = np.linspace(-2, 6, N)
    np.random.shuffle(x)
    y = 0.5 * x + 2 + 0.15 * np.random.randn(N,)
    return x,y

def createModel():
    model = Sequential()
    model.add(Dense(units=1, input_dim=1))
    model.compile(loss='mse', optimizer='sgd')
    return model

def plotRes(x_test,y_test,y_prediction):
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_prediction,color='r')
    plt.show()

def main():
    x,y = prepareDataSet()
    model = createModel()
    # train the first 160 data
    x_train, y_train = x[0:160], y[0:160]

    if 1:
        #start training
        model.fit(x_train, y_train, epochs=100, batch_size=64)
    else:
        for step in range(0, 100):
            cost = model.train_on_batch(x_train, y_train)
            if step % 20 == 0:
                print('cost is %f' % cost)

    # test on the rest 40 data
    x_test, y_test = x[160:], y[160:]

    # start evaluation
    cost_eval = model.evaluate(x_test, y_test, batch_size=40)
    print('evaluation lost %f' % cost_eval)

    model.summary()

    w, b = model.layers[0].get_weights()
    print('weight %f , bias %f' % (w, b))
    
    # start prediction
    y_prediction = model.predict(x_test)
    plotRes(x_test,y_test,y_prediction)
    pass

if __name__=='__main__':
    main()
    