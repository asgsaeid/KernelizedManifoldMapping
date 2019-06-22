import sys
from keras import backend as K
from keras.initializers import RandomUniform, Initializer, Constant
from keras.layers import Layer
from keras.layers import activations
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
import numpy as np

axis = -1
epsilon = 1e-3
kernel_initializer = initializers.get('glorot_uniform')
bias_initializer = initializers.get('zeros')

class PsiInitializer(Initializer):
    """
    Initializing as an identity matrix.
    """
    def __init__(self, diag = 1.0):
        self.diag = diag

    def __call__(self, shape, dtype=None):
        assert shape[0] == shape[1]
        return np.eye(shape[0], shape[1])

class kernelized_mapper(Layer):
    """ Layer of kernelized (RBF) mapper.

    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas

    """
    def __init__(self, output_dim, initializer=None, betas=2.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:  # if initializer is not specified, select the centers from the dataset
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(kernelized_mapper, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[2], input_shape[2]),
            initializer=kernel_initializer,
            name='kernel',
            trainable=True)

        self.bias = self.add_weight(
            shape=(input_shape[2],), 
            initializer=bias_initializer, 
            name='bias', 
            trainable=True)

        self.centers = self.add_weight(
            shape=(input_shape[2], input_shape[2]), 
            initializer=self.initializer, 
            name='centers', 
            trainable=True)
        
        self.betas = self.add_weight(
            shape=(input_shape[2],), 
            initializer=Constant(value=self.init_betas), 
            name='betas', 
            trainable=True)
        
        self.Psi = self.add_weight(
            shape=(input_shape[2], input_shape[2]), 
            initializer=PsiInitializer(diag=self.init_betas), 
            name='transformation', 
            trainable=True)

    def call(self, conv_features):
        def distance(conv_features, y, t_matrix):
            t_matrix = K.transpose(t_matrix) * t_matrix
            s = K.transpose(K.transpose(conv_features) - y)
            return K.sqrt(K.sum(K.dot(s, t_matrix) * s, axis=-1))

        expanded_centers = K.expand_dims(self.centers)
        expanded_centers = K.expand_dims(expanded_centers)

        h = distance(conv_features, expanded_centers, self.Psi + sys.float_info.epsilon)
        mah_like = K.exp(-self.betas * h)

        out = K.dot(mah_like, self.kernel)
        out_b = K.bias_add(out, self.bias, data_format=K.image_data_format())

        return out_b
    
    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[1], input_shape[2]

    def get_config(self):
        '''
        Define get_config() to be able to use model_from_json
        '''
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(kernelized_mapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
