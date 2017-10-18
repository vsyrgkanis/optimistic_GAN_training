from __future__ import print_function

from collections import defaultdict
from six.moves import range

from keras.engine.topology import Layer
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, Lambda
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from os.path import join,exists

#np.random.seed(1337)

K.set_image_data_format('channels_first')

class OneBias(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        super(OneBias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                      shape=(self.output_dim,),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(OneBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.bias_add(x, self.bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class OneWeight(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        super(OneWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='weight',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(OneWeight, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class polynomial(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1
        self.order = 4
        super(polynomial, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernels = []
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='glorot_normal',
                                    trainable=True)
        for o in range(self.order):
            self.kernels.append(self.add_weight(name='weight'+str(o),
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True))
        super(polynomial, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        t_x = x / 10
        for o in range(self.order):
            if o == 0:
                out = K.dot(t_x, self.kernels[o])
            else:
                out += K.dot(t_x, self.kernels[o])
            t_x *= x/10
        return K.bias_add(out, self.bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def D_act(x):
    return 2 - K.sigmoid(K.abs(x)) * 2

def D_act_shape(input_shape):
    return input_shape

def build_generator(latent_size, seqlen, nchannel):
    model = Sequential()
    #model.add(Dense(5, activation='tanh', input_shape=(1,)))
    #model.add(Dense(5, activation='tanh'))
    #model.add(Dense(1, input_shape=(1,)))
    model.add(OneBias(input_shape=(1,), name='G'))
    #model.add(OneWeight(input_shape=(1,), name='G'))
    #model.add(Lambda(generator_op, output_shape=generator_shape, input_shape=(1,)))
    return model

def build_discriminator(seqlen, nchannel, output_activation = 'sigmoid'):
    input_var = Input(shape=(1,))
    if output_activation == 'sigmoid':
       output = OneBias(name='D')(input_var)
       output = Lambda(D_act, output_shape=D_act_shape)(output)
    else:
       #output = Dense(5, activation='tanh')(input_var)
       #output = Dense(5, activation='tanh')(output)
       #output = Dense(5, activation='tanh')(output)
       #output = Dense(1, name='D')(output)
       #output = OneBias(name='D')(input_var)
       #output = OneWeight(name='D')(input_var)
       output = polynomial(name='D')(input_var)
    model = Model(inputs=input_var, outputs=output)
    return model
