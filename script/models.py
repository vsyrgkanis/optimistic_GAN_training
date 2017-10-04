from __future__ import print_function

from collections import defaultdict
from six.moves import range

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from os.path import join,exists

#np.random.seed(1337)

K.set_image_data_format('channels_first')

def build_generator(latent_size, seqlen, nchannel):
    model = Sequential()
    model.add(Dense(input_dim=latent_size, output_dim=128))
    model.add(Activation('tanh'))
    model.add(Dense(16*1*(seqlen/2)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((16,1,(seqlen/2)), input_shape=(16*1*(seqlen/2),)))
    model.add(UpSampling2D(size=(1,2), dim_ordering="th"))
    model.add(Convolution2D(nchannel,1,3, border_mode='same', dim_ordering="th"))
    model.add(Activation('tanh'))
    return model

def build_discriminator(seqlen, nchannel, output_activation = 'sigmoid'):
    model = Sequential()
    model.add(Convolution2D(16,1,3,
                            border_mode='same',
                            input_shape=(nchannel,1,seqlen),
                            dim_ordering="th"))
    model.add(Activation('tanh'))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation(output_activation))
    return model
