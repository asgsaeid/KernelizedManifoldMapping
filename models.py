import keras
from keras.engine import Input, Model
from keras.layers import Conv2D, Reshape, Flatten, Dense, concatenate

from mapper import kernelized_mapper

def ThreeLayersCNN(input_shape=(28,28,1), nb_filters=16, nb_classes=10, print_model_summary=False):
    '''
    Returns a 3-layered CNN Keras model with specified input shape and number of classes.
    '''
    input_layer = Input(shape=input_shape)

    conv1 = Conv2D(nb_filters, (8, 8), strides=(2, 2), activation='relu',padding='same')(input_layer)
    reshape1_1 = Reshape((14*14, nb_filters))(conv1)
    mapped1 = kernelized_mapper(nb_filters, betas=2.)
    kernel_mapper1 = mapped1(reshape1_1)
    reshape1_2 = Reshape((14, 14, nb_filters))(kernel_mapper1)

    conv2 = Conv2D(nb_filters*2, (6, 6), strides=(2, 2), activation='relu',padding='valid')(concatenate([conv1, reshape1_2]))
    reshape2_1 = Reshape((5*5, nb_filters*2))(conv2)
    mapped2 = kernelized_mapper(nb_filters*2, betas=2.)
    kernel_mapper2 = mapped2(reshape2_1)
    reshape2_2 = Reshape((5, 5, nb_filters*2))(kernel_mapper2)

    conv3 = Conv2D(nb_filters*2, (5, 5), strides=(1, 1), activation='relu',padding='valid')(concatenate([conv2, reshape2_2]))
    reshape3_1 = Reshape((1*1, nb_filters*2))(conv3)
    mapped3 = kernelized_mapper(nb_filters*2, betas=2.)
    kernel_mapper3 = mapped3(reshape3_1)
    reshape3_2 = Reshape((1, 1, nb_filters*2))(kernel_mapper3)

    flatten = Flatten()(concatenate([conv3, reshape3_2]))

    softmax = Dense(nb_classes, activation="softmax")(flatten)

    model = Model(inputs=input_layer, outputs=softmax)
    
    if print_model_summary:
        print(model.summary())

    return model
