#python3 tensorflow2 Steven 
#tensorflow backend K Test
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


#-----------simple descrpition of backend function--------------#
#reference: https://www.tensorflow.org/api_docs/python/tf/keras/backend
#abs():
#   abs all elements
#all():
#   logical AND
#any():
#    logical OR
#arange():
#   Creates a 1D tensor containing a sequence of integers.
#argmax():
#   Returns the index of the maximum value along an axis
#argmin():
#   Returns the index of the minimum value along an axis.
#backend():
#   Publicly accessible method for determining the current backend.
#mean()
#   Mean of a tensor, alongside the specified axis.
#dot():
#   Multiplies 2 tensors (and/or variables) and returns a tensor.
#eye():
#   Instantiate an identity matrix and returns it.
#shape():
#   Returns the symbolic shape of a tensor or variable.
#transpose():
#   Transposes a tensor and returns it.
#clip():
#   Element-wise value clipping.

def testBackEnd():
    x = tf.zeros([3, 4], tf.int32)
    print('x=',x)
    x = tf.zeros((3, 4), tf.int32)
    print('x=',x)
    return 
    
    a = tf.constant([1,2,3,4,5,6,7,8],dtype=tf.float32)
    a = K.reshape(a,(4,4))
    print('a=',a)

    a = tf.constant([[1, 2], [3, 4]],dtype=tf.float32)
    #a = K.abs(-1)
    print('a=',type(a),a) #a= tf.Tensor(1, shape=(), dtype=int32)
    # a = a.numpy()
    # print('a=',a)
    
    # a = tf.zeros([0, 3])
    # a = tf.concat([a, [[1, 2, 3], [5, 6, 8]]], axis=0)
    # print('a=',type(a),a)
    
    b = tf.constant([[1, 8], [2, 3]],dtype=tf.float32)
    
    c = K.square(a - b)
    print('c=',c)
    d = K.sum(c,axis=0)
    print('d=',d)
    d = K.sum(c,axis=1)
    print('d=',d)
    
    d = K.sum(c,axis=[0,1])
    print('d=',d)
    return

    a = K.abs([-1,0,9,-10])
    print('a=',a) #a= tf.Tensor([ 1  0  9 10], shape=(4,), dtype=int32)
    
    a = K.abs(np.array([-1,0,9,-10]))
    print('a=',a) #a= tf.Tensor([ 1  0  9 10], shape=(4,), dtype=int32)

    a = K.all(np.array([-1,0,9,-10]),axis=0)
    print('a=',a)  #a= tf.Tensor(False, shape=(), dtype=bool)

    a = K.all(np.array([[-1,-2,-1],
                        [ -1,0,9]]),axis=0) #x axis
    print('a=',a) #a= tf.Tensor([ True False  True], shape=(3,), dtype=bool)
    a = K.all(np.array([[-1,-2,-1],
                        [ -1,0,9]]),axis=1) #y axis
    print('a=',a) #a= tf.Tensor([ True False], shape=(2,), dtype=bool)
    
    a = K.arange(1,100,10)
    print('a=',a) #a= tf.Tensor([ 1 11 21 31 41 51 61 71 81 91], shape=(10,), dtype=int32)
    
    a = K.sum(np.array([-1,0,9,-10]))
    print('a=',a)#a= tf.Tensor(-2, shape=(), dtype=int32)
    
    a = K.square(np.array([-1,0,9,-10]))
    print('a=',a)#a= tf.Tensor([  1   0  81 100], shape=(4,), dtype=int32)
    
    x = K.placeholder(shape=(2, 3))
    y = K.placeholder(shape=(3, 4))
    xy = K.dot(x,y)
    shape = K.int_shape(xy)
    print('xy=',xy) #xy= Tensor("MatMul:0", shape=(2, 4), dtype=float32)
    print('xy shape=',shape) #xy shape= (2, 4)
    
    kvar = K.eye(3)
    #K.eval(kvar)
    print('kvar=',kvar)
    '''
    array([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]], dtype=float32)>
    '''
    
    a = np.array([[1,2],[3,4]])
    a = K.transpose(a)
    print('a=',a)
    '''
    a= tf.Tensor(
            [[1 3]
            [2 4]], shape=(2, 2), dtype=int32)
    '''
    
    a = K.clip(np.array([-1,0,1,2,3,4,5]), min_value=0, max_value=3)
    print('a=',a)#a= tf.Tensor([0 0 1 2 3 3 3], shape=(7,), dtype=int32)
    
    
def testConv():
    x_in = np.array([[1,2,3,4,5],
                     [6,7,8,9,10],
                     [2,3,4,5,6],
                     [5,6,7,8,9],
                     [4,5,6,3,2]])
    kernel_in = np.array([[1,2],
                        [3,4]])

    x = tf.constant(x_in.reshape(-1,5,5,1), dtype=tf.float32)
    kernel = tf.constant(kernel_in.reshape(2,2,1,1), dtype=tf.float32)
    
    print('x.shape=',x.shape)
    print('kernel.shape=',kernel.shape)
    y = tf.nn.conv2d(x, kernel, strides=[1], padding='VALID')
    
    print('y.shape=',y.shape)
    print('y=',y)

    #y_GT = tf.zeros(shape=(100,100), dtype=tf.float32)
    #y_GT = tf.Variable(0, shape=(200,200))
    y_GT = tf.Variable(tf.constant(0.0, shape = [200,200]))
    print('y_GT.shape=',y_GT.shape)
    #y_GT[1,2].assign(2)
    print('y_GT[1,2]=',y_GT[1,2])
    #y_GT[1,2] = 1
    y_GT[1,2].assign(1)
    print('y_GT[1,2]=',y_GT[1,2])
    print('y_GT=',y_GT)
    
def main():
    #testBackEnd()
    testConv()
    
if __name__=='__main__':
    main()
    