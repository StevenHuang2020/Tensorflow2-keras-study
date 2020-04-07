import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def splitData(X,y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    return x_train, x_test, y_train, y_test

def getCsvDataset(file,skipLines=3):
    df = pd.read_csv(file, header=None,skiprows=skipLines)
    #print(df.describe().transpose())
    #print(df.head())

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return splitData(X,y)

def getModelWeights(model,layer=0):
    for i in range(len(model.layers)):
        if layer != None:
            if layer != i:
                continue

        weights = model.layers[i].get_weights()
        return weights[0],weights[1]

def printModelWeights(model,layer=0):
    print("*" * 50,'model paramters')
    print('layers number=', len(model.layers))
    for i in range(len(model.layers)):
        if layer != None:
            if layer != i:
                continue

        weights = model.layers[i].get_weights()
        print('layer',i,'-------------len:',len(weights))
        print('weight shape:', weights[0].shape, 'b shape:', weights[1].shape)
        print('weight:', weights[0], 'b:',weights[1])
        #for j in range(len(weights)):
        #   print('weight',j,':',weights[j], end='')
        #print('')
        
def main():
    file = r'./db/fucDatasetReg_1F.csv'
    getCsvDataset(file)
    pass

if __name__ == "__main__":
    main()
