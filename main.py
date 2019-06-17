from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # Specify GPU IDs, select -1 for CPU

import keras
from keras.datasets import mnist
from keras import backend as K

from models import ThreeLayersCNN
from data import preprocess_data

# Training parameters
batch_size = 32
epochs = 12
num_filters = 16

# Dataset parameters
img_rows, img_cols = 28, 28 #input image dimensions for the MNIST dataset
num_classes = 10    #number of classes in the MNIST dataset

# Load the data, and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Specify channel order
K.set_image_data_format('channels_last')

# Preprocess the data
input_shape, x_train, y_train, x_test, y_test = preprocess_data(
    x_train, y_train, x_test, y_test, img_rows, img_cols, num_classes, 
    K.image_data_format(), print_shape=False
)

model = ThreeLayersCNN(
    input_shape=input_shape, nb_filters=num_filters, nb_classes=num_classes, 
    print_model_summary=False
)

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

model.fit(
    x_train, y_train, 
    batch_size=batch_size, 
    epochs=epochs, 
    verbose=1, 
    validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
