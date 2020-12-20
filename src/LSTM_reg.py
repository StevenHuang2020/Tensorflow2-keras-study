#python3 steven 
#LSTM regression, solve data set with time change

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

scaler = MinMaxScaler(feature_range=(0, 1))

def plotDataSet(data):
    plt.plot(data)
    plt.show()
    
def preprocessDb(dataset):
    dataset = scaler.fit_transform(dataset)
    print('dataset=',dataset[:5])
    return dataset

def splitData(dataset):
    # pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
    # print('pred_train.shape = ', pred_train.shape)
    # print('tar_train.shape = ', tar_train.shape)
    # print('pred_test.shape', pred_test.shape)
    # print('tar_test.shape', tar_test.shape)
    
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))
    return train,test

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def getDataSet():
    dataset = pd.read_csv('./db/airline-passengers.csv', usecols=[1])
    print(dataset.describe().T)
    print(dataset.head())
    print(dataset.shape)
    print(dataset.dtypes)
    db = dataset.values
    db = db.astype('float32')
    print('db.shape=',db.shape)
    #print('db=',db[:5])
    db = preprocessDb(db)
    return db
    
def createModel(look_back = 1):
    model = Sequential()
    model.add(LSTM(2, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    return model

def main():
    # fix random seed for reproducibility
    np.random.seed(7)
    dataset = getDataSet()
    train,test = splitData(dataset)

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    print(trainX[:5])
    #print(trainY[:5])
    
    #X = array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    #X_train = X.reshape(1, 3, 3) # X.reshape(samples, timesteps, features)
    print('trainX.shape = ',trainX.shape)
    print('trainY.shape = ',trainY.shape)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print('trainX.shape = ',trainX.shape)
    print('trainY.shape = ',trainY.shape)
    
    model = createModel(look_back)
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
    
    a = np.array([100.0]).reshape(1,1,1)
    print(a)
    print(model.predict(a))
    
    print('train5=',trainX[:5])
    trainPredict = model.predict(trainX[:5])
    print('pred5=',trainPredict)
    if 1:
        '''-----------start------evaluate-----'''
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
        
        print(len(trainPredictPlot), trainPredictPlot)
        print(len(testPredictPlot), testPredictPlot)
        
        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(dataset),label='dataset')
        plt.plot(trainPredictPlot,label='predictTrain')
        plt.plot(testPredictPlot,label='predictTest')
        plt.legend()
        plt.show()

if __name__=='__main__':
    main()