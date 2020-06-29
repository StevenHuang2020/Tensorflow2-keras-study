# python3 Steven GAN mnist test
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import argparse
from mnist_GAN import BATCH_SIZE,startToTrain
import PIL
import glob
import imageio
from IPython import display
import IPython

def getMnistNumber(train_images, train_labels,num=8):
    print(train_labels[:5])
    train_images = train_images[np.where(train_labels == num)]
    print('train_images.shape=',train_images.shape)
    return train_images
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help = 'epochs')
    return parser.parse_args()

def getMnistNumber(train_images, train_labels,num=8):
    print(train_labels[:5])
    train_images = train_images[np.where(train_labels == num)]
    print('train_images.shape=',train_images.shape)
    return train_images

def getData():
    global BATCH_SIZE
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    (train_images, train_labels), (_, _) = mnist.load_data()
    print('train_images.shape=',train_images.shape)
    print('train_labels.shape=',train_labels.shape)
    
    train_images = getMnistNumber(train_images, train_labels)
    
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    print('train_images.shape=',train_images.shape)
    BUFFER_SIZE = train_images.shape[0]
    
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print('train_dataset=',type(train_dataset))
    return train_dataset, BUFFER_SIZE//BATCH_SIZE

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

def gernerateGIF():
    anim_file = 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(r'.\images\image*.png')
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    if IPython.version_info > (6,2,0,''):
        display.Image(filename=anim_file)
  
def train():
    arg = argCmdParse()
    
    epochs = 50
    if arg.epoch:
        epochs = int(arg.epoch)
    print('epochs=',epochs)
    
    train_dataset,batches = getData()
    startToTrain(train_dataset,epochs,batches)
    
def main():
    train()
    #gernerateGIF()

if __name__=='__main__':
    main()

   