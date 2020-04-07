import matplotlib.pyplot as plt
import numpy as np 

def plotLossAx(loss,ax,label='Loss'):
    ax.plot(loss, label=label)
    ax.legend()

def plotLoss(loss,name='Loss'):
    plt.title(name)
    plt.plot(loss)
    plt.show()

def plotLosses(loss1,loss2,name='Losses'):
    plt.title(name)
    plt.plot(loss1)
    plt.plot(loss2)
    plt.show()

def plotLosseList(losses,labels):
    for i,loss in enumerate(losses):
        plt.plot(loss,label=labels[i])
    plt.show()

def plotLosseListTu(losseLabels): #loss label tuple
    for i in losseLabels:
        plt.plot(i[0],label=i[1])
    plt.legend()
    plt.show()

def plotLossAndAcc(loss,acc,name='Loss&Acc'):
    plt.title(name)
    plt.plot(loss, label='loss')
    plt.plot(acc, label='accuracy')
    plt.legend()
    plt.show()

def plotSubLossAndAcc(loss,acc,name='Loss&Acc'):
    plt.title(name)
    ax = plt.subplot(1,2,1)
    ax.plot(loss, label='loss')
    ax.legend()
    
    ax = plt.subplot(1,2,2)
    ax.plot(acc, label='accuracy',color='g')
    ax.legend()

    plt.show()
    
def plotResultError(scores_train,scores_test,loss):
    error_train = np.array(1-np.array(scores_train))
    #print(error_train)
    error_test = np.array(1 - np.array(scores_test))

    plt.title('Error&Loss over epochs use MLP')
    plt.plot(error_train, label='error_train')
    #plt.plot(error_test, label='error_test')
    plt.plot(loss, label='loss')
    plt.legend()
    plt.show()

def plotResult(scores_train,scores_test,loss):
    if 0: #style 1
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        ax[0].plot(scores_train)
        ax[0].set_title('Train mean accuracy')
        ax[1].plot(scores_test)
        ax[1].set_title('Test mean accuracy')
        ax[2].plot(loss)
        ax[2].set_title('Loss')
        fig.suptitle("Accuracy &Loss over epochs", fontsize=14)
    else:# style 2
        plt.title('Acc&Loss over epochs use MLP')
        plt.plot(scores_train, label='scores_train')
        plt.plot(scores_test, label='scores_test')
        plt.plot(loss, label='loss')
        plt.legend()
    plt.show()

def plotTrainRes(x_train, y_train, x_test, y_test, loss):
    ax = plt.subplot(1, 2, 1)
    ax.title.set_text('training dataset')
    plt.scatter(x_train[:,0], y_train, label='input dataset')  # plot train dataset

    plt.plot(x_test[:,0],y_test,color='r')

    ax = plt.subplot(1, 2, 2)
    ax.title.set_text('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.show()

def test6Plot():
    # createdata
    N = 100
    x_data = np.random.rand(N)
    y_data = np.random.rand(N)
    #plt.show(scatterplot(x_data, y_data,yscale_log=False))

    #plt.show(lineplot(x_data, y_data, x_label = "x_label", y_label="y_label", title="title"))
    plt.show(histogram(x_data, n_bins=10))
    pass


if __name__ == "__main__":
    test6Plot()