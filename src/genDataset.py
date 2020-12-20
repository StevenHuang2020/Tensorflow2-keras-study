"""
Steven 20/03/2020
generate dataset
"""
#python3 steven
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gSavePath=r'./db/'

def noise(len):
    return np.random.rand(len)
    
def funN(x1,x2):
    return 2.2*x1 + 0.8*x2 + noise(len(x1))*2
    
def funN2(x1,x2,x3):
    return 2.2*x1 + 0.8*x2 -1.7*x3 + noise(len(x1))*2

def appendComment(file,comment,mode='a'):
    with open(file, mode) as f:
        f.write(comment)

def createCSV_NFeatures(Num=100):
    X0 = 2*np.random.randn(Num)+2     
    X0 = np.around(X0,decimals=2)
    X1 = 2.3*np.random.randn(Num)+1.6
    X1 = np.around(X1,decimals=2)
    X2 = 2.8*np.random.randn(Num)+1.8
    X2 = np.around(X2,decimals=2)

    #print(X0.shape)
    #print(X1.shape)
   
    #y = funN(X0,X1)
    y = funN2(X0,X1,X2)
    y = np.around(y,decimals=2)
    #print('X0.shape = ', X0.shape)
    #print('y.shape = ', y.shape)
    X0 = X0.reshape((X0.shape[0],1))
    X1 = X1.reshape((X1.shape[0],1))
    X2 = X2.reshape((X2.shape[0],1))
    y = y.reshape((y.shape[0],1))

    if 0:
        file = gSavePath + 'fucDatasetReg_2F.csv'
        appendComment(file,'','w')
        appendComment(file,'#regression dataset two feature linear\n')
        appendComment(file,'#y = 2.2*x1 + 0.8*x2 + noise\n')

        f = np.hstack((X0,X1))
        f = np.hstack((f,y))
        print('y.shape = ', y.shape)
        df = pd.DataFrame(f)
        print(df.shape)
        df.to_csv(file,index=False,header=['x0','x1','y'],mode='a')
    elif 0:
        file = gSavePath + 'fucDatasetReg_3F.csv'
        appendComment(file,'','w')
        appendComment(file,'#regression dataset three feature\n')
        appendComment(file,'# y = 2.2*x1 + 0.8*x2 -1.7*x3 + noise\n')

        f = np.hstack((X0,X1))
        f = np.hstack((f,X2))
        f = np.hstack((f,y))
        print('y.shape = ', y.shape)
        df = pd.DataFrame(f)
        print(df.shape)
        df.to_csv(file,index=False,header=['x0','x1','x2','y'],mode='a')
    elif 0:
        file = gSavePath + 'fucDatasetClf_2F_M.csv'
        appendComment(file,'','w')
        appendComment(file,'#classifier dataset two feature,multify class\n')
        appendComment(file,'# 1.6*x0 + 0.8*x1>0 ? 1: 0\n')

        a = X0*2.2+0.8*X1+3.2
        #print(np.mean(a),np.min(a),np.max(a)) #-2 8 20
        
        y = np.where(X0*2.2+0.8*X1 > 0, 1, 0)
           
        f = np.hstack((X0,X1))
        f = np.hstack((f,y))
        print('y.shape = ', y.shape)
        df = pd.DataFrame(f)
        print(df.shape)
        df.to_csv(file,index=False,header=['x0','x1','y'],mode='a')
    elif 1:
        file = gSavePath + 'fucDatasetClf_2F_MClass.csv'
        appendComment(file,'','w')
        appendComment(file,'#classifier dataset two feature,multify class\n')
        appendComment(file,'# 2.2*x0 + 0.8*x1 + 3.2  (<2 : 0, ( >=2 and <=8 ):1 , >8 : 2 ) \n')

        a = X0*2.2+0.8*X1+3.2
        print(np.mean(a),np.min(a),np.max(a)) #-2 8 20
        
        y = np.zeros((len(X0),1),dtype=np.int32)
        y[np.where(a<=2)[0]]=0
        y[np.where(a>2)[0]]=1
        y[np.where(a>8)[0]]=2
        #print(y[:20])
   
        f = np.hstack((X0,X1))
        f = np.hstack((f,y))
        print('y.shape = ', y.shape)
        df = pd.DataFrame(f)
        print(df.shape)
        df.to_csv(file,index=False,header=['x0','x1','y'],mode='a')
    pass

def createCSV(Num=100, gamma=0.01):
    def fuc(x):
        return 2.2*x + 3.8

    def fucNoLinear(x):
        return 0.2*x**2 + 3.2*x + 0.9
    
    X0 = np.linspace(-2, 5, Num)
    X0 = np.around(X0, decimals=2)
    
    if 0:
        y = fuc(X0) + np.around(noise(Num)*gamma, 2)
        #y = np.around(y,decimals=2)
    
        file = gSavePath + 'fucDatasetReg_1F.csv'
        appendComment(file, '', 'w')
        appendComment(file, '#regression dataset one feature\n')
        appendComment(file, f'#y = 2.2*x + 3.8 + noise*{gamma}\n')
    
    else:
        y = fucNoLinear(X0) + np.around(noise(Num)*gamma, 2)

        file = gSavePath + 'fucDatasetReg_1F_NoLinear.csv'
        appendComment(file, '', 'w')
        appendComment(file, '#regression dataset one feature no linear\n')
        appendComment(file, f'#y = 0.2*x**2 + 3.2*x + 0.9 + noise*{gamma}\n')

    X0 = X0.reshape((X0.shape[0],1))
    y = y.reshape((y.shape[0],1))
    df = pd.DataFrame(np.hstack((X0,y)))
    df.to_csv(file,index=False,header=['x','y'],mode='a')
    
    print('X0.shape = ', X0.shape)
    print('y.shape = ', y.shape)
    
def main():
    createCSV()
    #createCSV_NFeatures(1000)
    
if __name__=='__main__':
    main()
