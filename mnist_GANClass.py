# python3 Steven GAN mnist test
import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras.datasets import mnist
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from IPython import display
from progressBar import SimpleProgressBar
import argparse
import datetime

BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 16
    
def argCmdParse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', help = 'epochs')
    return parser.parse_args()

def getData():
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    (train_images, train_labels), (_, _) = mnist.load_data()
    print('train_images.shape=',train_images.shape)
    print('train_labels.shape=',train_labels.shape)
    
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    print('train_images.shape=',train_images.shape)
    BUFFER_SIZE = train_images.shape[0]
    #BUFFER_SIZE = 60000
    
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print('train_dataset=',type(train_dataset))
    return train_dataset, BUFFER_SIZE//BATCH_SIZE

class Mnist_GAN():
    def __init__(self):
        self.InitTrain()
    
    def make_generator_model(self):
        print('-----------------generator_model-------------------')
        model = tf.keras.Sequential()
        if 1:
            model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Reshape((7, 7, 256)))
            assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

            model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            assert model.output_shape == (None, 7, 7, 128)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            assert model.output_shape == (None, 14, 14, 64)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
            assert model.output_shape == (None, 28, 28, 1)
        else:
            model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(100,)))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

            model.add(layers.Reshape((4, 4, 256)))
            assert model.output_shape == (None, 4, 4, 256) # Note: None is the batch size

            model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            assert model.output_shape == (None, 4, 4, 128)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            
            model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            assert model.output_shape == (None, 8, 8, 64)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())
            
        model.summary()
        return model
        
    def make_discriminator_model(self):
        print('-----------------discriminator_model-------------------')
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        model.summary()
        return model

    #Discriminator loss
    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    #Generator loss
    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def generate_and_save_images(self, epoch):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(self.seed, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(r'./images/image_at_epoch_{:04d}.png'.format(epoch))
        #plt.show()

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        #print('noise.shape=',noise.shape,type(noise))
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss,disc_loss

    def train(self, dataset, epochs, batches):
        total = batches
        pb = SimpleProgressBar(total=total)
        for epoch in range(epochs):
            start = time.time()

            i=0
            for image_batch in dataset:
                genLoss,discLoss = self.train_step(image_batch)
                i = i+1
                pb.update(i)
                #print('Run/total: {}/{},percent:{}%'.format(i,total,round(i*100/total,3))) 
                #break
            
            if epoch % 10 == 0:
                self.generate_and_save_images(epoch + 1)

            # Save the model every 15 epochs
            if epoch % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            #print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            print ('Time for epoch {}/{} is {} sec,genLoss:{},discLoss:{}'.format(epoch + 1, epochs, round(time.time()-start,4),genLoss,discLoss))

        # Generate after the final epoch
        self.generate_and_save_images(epochs)

    def InitTrain(self):
        self.generator = self.make_generator_model()
        #noise = tf.random.normal([1, 100])
        #generated_image = generator(noise, training=False)
        #print('generated_image.shape=',generated_image.shape)
        #showImage(generated_image[0, :, :, 0])
        
        self.discriminator = self.make_discriminator_model()
        #decision = discriminator(generated_image)
        #print (decision)
        
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        
        # We will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)
        self.seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    def startToTrain(self,train_dataset,epochs,batches):
        #self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.train(train_dataset, epochs,batches)

def showImage(img):
    plt.imshow(img, cmap='gray')
    plt.show()
        
def testGAN():
    model = Mnist_GAN()
    #model.checkpoint.restore(tf.train.latest_checkpoint(model.checkpoint_dir))
    
    noise = tf.random.normal([1, 100])
    print('noise.shape=',noise.shape)
    generated_image = model.generator(noise, training=False)
    print('generated_image.shape=',generated_image.shape)
    showImage(generated_image[0, :, :, 0])
   
    #discriminator = make_discriminator_model()
    decision = model.discriminator(generated_image)
    print (decision)
    
def main():
    arg = argCmdParse()
    epochs = 50
    if arg.epoch:
        epochs = int(arg.epoch)
    print('epochs=',epochs)
    
    return testGAN()
    
    train_dataset, batches = getData()
    model = Mnist_GAN()
    #call after first training to continue training
    #model.checkpoint.restore(tf.train.latest_checkpoint(model.checkpoint_dir)) #not call when first training
    model.startToTrain(train_dataset,epochs,batches)

if __name__=='__main__':
    main()
    #plotLoss(r'.\images\ganLog.txt')
    
def plotLoss(log_file,name='Training loss'):    
    def getLoss(log_file,startIter=0,stopIter=None):
        numbers = {'1','2','3','4','5','6','7','8','9'}
        with open(log_file, 'r') as f:
            lines  = [line.rstrip("\n") for line in f.readlines()]
            
            epochs = []
            genLoss = []
            discLoss=[]
            for line in lines:
                trainIterRes = line.split(' ')
                
                epochs.append(int(trainIterRes[3]))
                #print(trainIterRes[6])
                trainIterRes = trainIterRes[6].split(',')
                #print(trainIterRes[1],trainIterRes[2])
                
                gen = trainIterRes[1]
                gen = float(gen[gen.find(':')+1 : ])
                disc = trainIterRes[2]
                disc = float(disc[disc.find(':')+1 : ])
                #print(gen,disc)
                genLoss.append(gen)
                discLoss.append(disc)
                
        return epochs,genLoss,discLoss

    epochs,genLoss,discLoss = getLoss(log_file)
    
    ax = plt.subplot(1,1,1)
    #ax.set_title(name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.plot(epochs,genLoss,label='generator Loss')
    ax.plot(epochs,discLoss,label='discriminator Loss')
    #plt.ylim(0, 4)
    #plt.yscale("log")
    plt.legend()
    plt.show()
    

   