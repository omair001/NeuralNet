# -*- coding: utf-8 -*-
"""
This project trains a convolutional neural network to detect human emotion after training it
from a dataset of images.

The dataset is preprocessed by splitting into the training set and the 
test set with an 80:20 ratio. given thousands of images of happy and sad people,
the neural network learns to recognize features that demonstrate those emotions.

this file contains the code that trains and saves a neural network


@authors: Bardia Goharanpour and Syed Omair Anwar

"""

# importing libraries 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
# this library modifies the input images so that different transformations 
# such as rotations and mirroring are added to the training
from keras.preprocessing.image import ImageDataGenerator

# create a CNN oc type Sequential
classifier = Sequential()

# layer for manipulating the input images to identify features in feature map 
# we will use 64 feature maps that will be 3x3 and the input shapes of the 
# images is 256x256 and we are giving it 3 channels for RGB colours
# the activation function is the rectifier function
classifier.add(Convolution2D(64, 3, 3, input_shape=(256, 256, 3), activation = 'relu'))
# the max pooling step using a 2x2 pooling filter compresses the feature map 
# to highlight the important features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# more layers are added to the neural network to improve feature detection and max pooling
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(256, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# keras flattens the layer of the CNN into a 1D vector ready to be used with an ANN
classifier.add(Flatten())

# we are making our ANN with one hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# compiling the neural network as done previously
# we are using binary cross entropy for the loss function to know the
# amounts the weights need to be modified by when backpropagating
classifier.compile(optimizer =  'adam', loss = 'binary_crossentropy', metrics =['accuracy'])

# This allows us to flip shear and manipulate the image to improve training
# with images that look different for the training set 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

# no manipulations done for the test set as we are using it for validation
test_datagen = ImageDataGenerator(rescale=1./255)

# directory for training set, resizes images to 128x128
training_set = train_datagen.flow_from_directory('dataset/trainset',
                                                 target_size=(256, 256),
                                                 batch_size=16,
                                                 class_mode='binary')

# directory for test set
test_set = test_datagen.flow_from_directory('dataset/testset',
                                            target_size=(256, 256),
                                            batch_size=16,
                                            class_mode='binary')



# this trains the CNN
classifier.fit_generator(training_set,
                         steps_per_epoch=(7993/16),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=(2000/16),
                         workers=12,
                         max_q_size=100)

classifier.save('expressions.h5')