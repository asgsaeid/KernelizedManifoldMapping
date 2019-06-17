import keras
from keras import backend as K

def preprocess_data(x_train, y_train, x_test, y_test, num_rows, num_cols, nb_classes, channel_order, print_shape=False):

    # Set channel order
    if channel_order == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, num_rows, num_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, num_rows, num_cols)
        input_shape = (1, num_rows, num_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], num_rows, num_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], num_rows, num_cols, 1)
        input_shape = (num_rows, num_cols, 1)

    # Preprocess the data
    x_train = K.cast_to_floatx(x_train)
    x_test = K.cast_to_floatx(x_test)
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    if print_shape:
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples.')
        print(x_test.shape[0], 'test samples.')

    return input_shape, x_train, y_train, x_test, y_test