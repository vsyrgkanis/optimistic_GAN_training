from __future__ import print_function

from collections import defaultdict
from six.moves import range

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input, Lambda
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from os.path import join,exists

#np.random.seed(1337)

K.set_image_data_format('channels_first')

def generator_op(x):
    weight = K.random_normal_variable(shape=(1,), mean=0, scale=1)
    return x + weight

def generator_shape(input_shape):
    return input_shape

def build_generator(latent_size, seqlen, nchannel):
    model = Sequential()
    model.add(Dense(5, activation='tanh', input_shape=(1,)))
    model.add(Dense(5, activation='tanh'))
    model.add(Dense(1))
    #model.add(Lambda(generator_op, output_shape=generator_shape, input_shape=(1,)))
    return model

def disc_op(x):
    weight = K.random_normal_variable(shape=(1,), mean=0, scale=1)
    #return 1 - K.sigmoid(K.abs(x - weight))
    diff = K.abs(x - weight)
    return 1/ (diff + 1)

def disc_shape(input_shape):
    return input_shape

def build_discriminator(seqlen, nchannel, output_activation = 'sigmoid'):
    input_var = Input(shape=(1,))
    if output_activation == 'sigmoid':
       output = Lambda(disc_op, output_shape=disc_shape)(input_var)
    else:
       output = Dense(5, activation='tanh')(input_var)
       output = Dense(5, activation='tanh')(output)
       output = Dense(5, activation='tanh')(output)
       output = Dense(1)(output)
    model = Model(inputs=input_var, outputs=output)
    return model
