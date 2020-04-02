#python3 steven tf 2.1.0
#01/04/2020
import tensorflow.keras as ks
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from mnistCnn import prepareMnistData


def createModel(input_shape, classes):
    model = Sequential()

    #output layer neurons shoule eqaul to the number of class
    #activation options: elu softmax selu softplus softsign relu tanh sigmoid hard_sigmoid exponential linear 
    if 0:    # 1 layer
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(classes, activation='softmax'))
    else:    # multify layers
        model.add(Conv2D(32, kernel_size=(3, 3), 
                    activation='relu', 
                    input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        #model.add(Conv2D(32, (3, 3), activation="relu"))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.5))
        
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax')) #can not give input_shape,auto match
  
     
    lr = 0.01
    opt = optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    #opt = optimizers.RMSprop(lr=0.001, rho=0.9)
    #opt = optimizers.Adagrad(learning_rate=lr)
    #opt = optimizers.Adadelta(learning_rate=lr, rho=0.95)
    #opt = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt = optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  #loss='mean_squared_error', #  #mean_absolute_error
                  #loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True), #number class
                  loss=ks.losses.categorical_crossentropy, #one hot class
                metrics=['accuracy'])
    
    model.summary()
    return model

def main():
    num_classes = 10
    x_train, y_train, x_test, y_test, input_shape = prepareMnistData(0.2)
    model = createModel(input_shape,num_classes)
    history = model.fit(x=x_train, y=y_train, epochs=5)
    #printModelWeights(model)
    
    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
    print('\nTest accuracy:', test_acc,'loss=',test_loss)

if __name__=='__main__':
    main()