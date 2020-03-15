#python3 steven tf 2.1.0
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class LinearModel:
    def __call__(self, x):
        return self.Weight * x + self.Bias

    def __init__(self):
        self.Weight = tf.Variable(11.0)
        self.Bias = tf.Variable(12.0)

def plotSubplot(x_train, y_train, w, b, epochs, loss):
    ax = plt.subplot(1, 2, 1)
    ax.title.set_text('training dataset')

    plt.scatter(x_train, y_train, label='input dataset')  # plot train dataset

    x = np.linspace(0, 3, 10)  # plot train result w & b
    y = w * x + b
    plt.plot(x, y, color='r')

    ax = plt.subplot(1, 2, 2)
    ax.title.set_text('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss)

    plt.show()


def plotSamplesAndPredict(x_train, y_train, w, b):
    plt.scatter(x_train, y_train, label='input dataset')  #plot train dataset

    x = np.linspace(0, 3, 10) #plot train result w & b
    y = w*x + b
    plt.plot(x, y, color='r')
    plt.show()

def plotSamples(x, y):
    plt.scatter(x, y, label='input dataset')
    plt.show()

def plotEpochAndLoss(x, y):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, y)
    plt.show()

def preparDataSet(N=50):
    mSamples = N
    X = np.linspace(0, 3, mSamples)
    noise = np.random.randn(X.shape[0]) * 0.3
    y = 2 * X + 0.9 + noise
    #print(x)
    #print(y)
    return X, y

def lossFuc(y, pred):
    return tf.reduce_mean(tf.square(y - pred))

def model(x, y):
    w = tf.Variable(0.0) #inital value
    b = tf.Variable(0.0)
    y_pred = w*x + by_pred
    return

def train(linear_model, x, y, lr=0.001):
    with tf.GradientTape() as t:
        current_loss = lossFuc(y, linear_model(x))

    lr_weight, lr_bias = t.gradient(current_loss, [linear_model.Weight, linear_model.Bias])
    linear_model.Weight.assign_sub(lr * lr_weight)
    linear_model.Bias.assign_sub(lr * lr_bias)

def main():
    print("*"*100)
    x, y = preparDataSet()
    lr = 0.1

    linear_model = LinearModel()
    Weights, Biases = [], [] #save all w & b in process
    epoch_countX = []
    real_lossY = []
    epochs = 100
    for epoch_count in range(epochs):
        Weights.append(linear_model.Weight.numpy())
        Biases.append(linear_model.Bias.numpy())
        real_loss = lossFuc(y, linear_model(x))
        train(linear_model, x, y, lr=lr)
        if epoch_count % 10 == 0:
            print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")
        epoch_countX.append(epoch_count)
        real_lossY.append(real_loss)

    w = linear_model.Weight.numpy()
    b = linear_model.Bias.numpy()
    print('Final weight:', w, 'Finale bias:', b)

    #plotSamples(x,y)
    #plotSamplesAndPredict(x, y, w, b)
    #plotEpochAndLoss(epoch_countX,real_lossY)
    plotSubplot(x, y, w, b, epoch_countX,real_lossY)

if __name__=='__main__':
    main()


