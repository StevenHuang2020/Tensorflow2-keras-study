from __future__ import print_function
import tensorflow.keras as ks
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import datetime

num_classes = 10

def prepareMnistData(nr=1.0):   #nr: ratio of sample numbers wanted
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print('Raw data set:')
    # print('X_train.shape = ', x_train.shape)
    # print('y_train.shape = ', y_train.shape)
    # print('X_test.shape = ', x_test.shape)
    # print('Y_test.shape = ', y_test.shape)
    # print('one sample=', x_train[0,:].shape)
    # print('fmt=',K.image_data_format())
    print('y_train[:5]=',y_train[:5])
    print('-'*20)
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = ks.utils.to_categorical(y_train, num_classes)
    y_test = ks.utils.to_categorical(y_test, num_classes)
    print('y_train[:5]=',y_train[:5])
    
    lTrain = int(x_train.shape[0]*nr)
    lTest = int(x_test.shape[0]*nr)
    x_train = x_train[:lTrain,:]
    y_train = y_train[:lTrain,:]
    x_test = x_test[:lTest,:]
    y_test = y_test[:lTest,:]
    print('Back data set:')
    print('X_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', x_test.shape)
    print('Y_test.shape = ', y_test.shape)
    return x_train, y_train, x_test, y_test, input_shape
    
def createModel(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    lr = 0.1
    opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)    
    #opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss=ks.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])
    
    model.summary()
    return model

def main():
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData()
    print('input_shape = ', input_shape)
    model = createModel(input_shape,num_classes)
    
    log_dir = r"logs\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = ks.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
    model.fit(x_train, y_train, epochs=5, callbacks = [tensorboard_callback])
    #model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    pass

if __name__=='__main__':
    main()