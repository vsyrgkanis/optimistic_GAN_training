#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle

from six.moves import range

from PIL import Image
from keras.datasets import mnist
import keras.backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, Lambda, Activation
from keras.models import Sequential, Model
from keras.utils.generic_utils import Progbar
import numpy as np, sys
from os.path import join,exists
from os import system, makedirs
import theano.tensor as T
import models as models
import utils as utils
from oftl import *

#np.random.seed(1337)

K.set_image_data_format('channels_first')

if __name__ == '__main__':

    datadir = sys.argv[1]
    outdir = sys.argv[2]
    seqlen = int(sys.argv[3])
    nchannel = int(sys.argv[4])
    optimizer = sys.argv[5]
    optimizer_lr = sys.argv[6]

    outdir = '_'.join([outdir,optimizer,str(optimizer_lr)])
    if exists(outdir):
        system('rm -r ' + outdir)
    makedirs(outdir)
    system('cp ' + __file__ + ' '+ outdir)
    system('cp ' + 'models.py' + ' '+ outdir)

    # batch and latent size taken from the paper
    epochs = 50
    batch_size = 128
    latent_size = 50
    numdisplay = 100000

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    #optimizer_lr = 5e-04
    optimizer_lr = float(optimizer_lr)
    optim_mapper = {
            'SGD':SGD,
            'RMSprop':RMSprop,
            'SGD_v1':SGD_v1,
            'RMSprop_v1':RMSprop_v1,
            'RMSprop_v2':RMSprop_v2,
            'RMSprop_v3':RMSprop_v3,
            'RMSprop_v4':RMSprop_v4,
            'SGD_v3':SGD_v3,
            'SGD_v4':SGD_v4,
            'SGD_v2':SGD_v2
            }
    d_optim = optim_mapper[optimizer](lr=optimizer_lr)
    g_optim = optim_mapper[optimizer](lr=optimizer_lr)

    # build the discriminator
    discriminator = models.build_discriminator(seqlen, nchannel)
    utils.set_trainability(discriminator, True)
    discriminator.compile(
        optimizer=d_optim,
        loss='binary_crossentropy'
    )

    # build the generator
    generator = models.build_generator(latent_size, seqlen, nchannel)
    generator.compile(optimizer=g_optim,
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))

    # get a fake image
    fake = generator(latent)

    # we only want to be able to train generation for the combined model
    utils.set_trainability(discriminator, False)
    fake = discriminator(fake)
    combined = Model(latent, fake)

    combined.compile(
        optimizer="SGD",
        loss='binary_crossentropy'
    )

    (X_train, y_train), (X_test, y_test) = utils.load_data(join(datadir,'embed/CV0'))
    X_train = (X_train.astype(np.float32) - 0.5) / 0.5
    X_test = (X_test.astype(np.float32) - 0.5) / 0.5

    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    #X_train = np.expand_dims(X_train, axis=1)

    #X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    #X_test = np.expand_dims(X_test, axis=1)

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)
    num_batches = int(X_train.shape[0] / batch_size)

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch + 1, epochs))

        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]

        progress_bar = Progbar(target=num_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            progress_bar.update(index)

            # Train D on real images
            utils.set_trainability(discriminator, True)

            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            y_batch = np.array([1] * batch_size)

            t_disc_loss = discriminator.train_on_batch(image_batch,y_batch)


            # Train D on fake images
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            generated_images = generator.predict(noise, verbose=0)
            y_generated = np.array([0] * batch_size)

            epoch_disc_loss.append(t_disc_loss +
                    discriminator.train_on_batch(generated_images, y_generated))


            # Train G
            utils.set_trainability(discriminator, False)

            noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
            trick =  np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(noise, trick))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # Evaluate the testing loss of D
        noise = np.random.normal(0, 1, (num_test, latent_size))

        generated_images = generator.predict(
            noise, verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * num_test + [0] * num_test)

        discriminator_test_loss = discriminator.evaluate(
            X, y, verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # Evaluate the testing loss of G
        noise = np.random.normal(0, 1, (2 * num_test, latent_size))

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            noise,
            trick, verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # Generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} '.format(
            'component', discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f}'
        print(ROW_FMT.format('generator (train)',
                             train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             test_history['discriminator'][-1]))

        # Save weights every epoch
        generator.save_weights(
            join(outdir,'params_generator_epoch_{0:03d}.hdf5'.format(epoch)), True)
        discriminator.save_weights(
            join(outdir,'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch)), True)

        # Generate some digits to display
        noise = np.random.normal(0, 1, (numdisplay, latent_size))

        generated_images = generator.predict(noise, verbose=0)
        pickle.dump(generated_images, open(join(outdir, 'plot_epoch_{0:03d}_generated.pkl'.format(epoch)),'wb'))
        # arrange them into a grid
        #img = (np.concatenate([r.reshape(-1, seqlen)
        #    for r in np.split(generated_images, 10)
        #        ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        #Image.fromarray(img).save(join(outdir, 'plot_epoch_{0:03d}_generated.png'.format(epoch)))

        # Dump history
        pickle.dump({'train': train_history, 'test': test_history},
                open(join(outdir,'acgan-history.pkl'), 'wb'))
