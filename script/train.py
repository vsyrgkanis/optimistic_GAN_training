#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
from six.moves import range
import numpy as np, sys, argparse, cPickle as pickle
from os.path import join,exists
from os import system, makedirs

from PIL import Image
from keras.datasets import mnist
import keras.backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input
from keras.models import Model
from keras.utils.generic_utils import Progbar
import theano.tensor as T
import models as models
import utils as utils
from oftl import *
from optimizer import *

K.set_image_data_format('channels_first')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='datadir')
    parser.add_argument("-o", dest='outdir')
    parser.add_argument("-l", dest='seqlen', type=int)
    parser.add_argument("-c", dest='nchannel', type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr", dest='optimizer_lr', type=float)
    parser.add_argument("-v", dest='version', type=int)
    parser.add_argument("-s", dest='schedule')
    parser.add_argument("--g_interval", dest='train_G_interval', type=int, default=1)
    parser.add_argument("-t", dest='network_type', default='GAN')
    parser.add_argument("-b", dest='batch_size', type=int, default=128)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    schedule = args.schedule if args.schedule != 'None' else None

    # Copy over arguments and model files
    outdir = '_'.join([args.outdir, args.optimizer, str(args.optimizer_lr)])
    if exists(outdir):
        system('rm -r ' + outdir)
    makedirs(outdir)
    system('cp ' + __file__ + ' '+ outdir)
    system('cp ' + 'models.py' + ' '+ outdir)
    with open(join(outdir, 'commandline_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

    # Parameters that matter less
    epochs = 50
    batch_size = args.batch_size
    latent_size = 50
    numdisplay = 100000

    # Optimizer
    optim_mapper = {
            'OFRL': OFRL,
            'OMDA': OMDA,
            }
    d_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, version=args.version, schedule=schedule)
    g_optim = optim_mapper[args.optimizer](lr=args.optimizer_lr, version=args.version, schedule=schedule)

    # Loss related (wgan is different from regular gan)
    if args.network_type not in ['wgan', 'gan']:
        raise ValueError('network_type {} not recognized'.format(args.network_type))
    loss_func = utils.modified_binary_crossentropy if args.network_type == 'wgan' else 'binary_crossentropy'
    fake_label = -1 if args.network_type == 'wgan' else 0
    output_activation = 'linear' if args.network_type == 'wgan' else 'sigmoid'

    # build the discriminator
    discriminator = models.build_discriminator(args.seqlen, args.nchannel, output_activation=output_activation)
    utils.set_trainability(discriminator, True)
    discriminator.compile(optimizer=d_optim,
                          loss=loss_func)

    # build the generator
    generator = models.build_generator(latent_size, args.seqlen, args.nchannel)
    generator.compile(optimizer='SGD',
                      loss='binary_crossentropy')

    # we only want to be able to train generation for the combined model
    latent = Input(shape=(latent_size, ))
    utils.set_trainability(discriminator, False)
    fake = generator(latent)
    fake = discriminator(fake)
    combined = Model(latent, fake)
    combined.compile(optimizer=g_optim,
                     loss=loss_func)

    # Load the data
    (X_train, y_train), (X_test, y_test) = utils.load_data(join(args.datadir, 'embed/CV0'))
    X_train = (X_train.astype(np.float32) - 0.5) / 0.5
    X_test = (X_test.astype(np.float32) - 0.5) / 0.5
    num_train, num_test = X_train.shape[0], X_test.shape[0]

    # Training
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
            y_generated = np.array([fake_label] * batch_size)
            epoch_disc_loss.append(t_disc_loss +
                    discriminator.train_on_batch(generated_images, y_generated))

            # Clip weights for WGAN
            if args.network_type == 'wgan':
                weights = [np.clip(w, -0.01, 0.01) for w in discriminator.get_weights()]
                discriminator.set_weights(weights)

            if (1 + index) % args.train_G_interval == 0:
                # Train G
                utils.set_trainability(discriminator, False)
                noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
                trick =  np.ones(2 * batch_size)
                epoch_gen_loss.append(combined.train_on_batch(noise, trick))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # Evaluate the testing loss of D
        noise = np.random.normal(0, 1, (num_test, latent_size))
        generated_images = generator.predict(noise, verbose=False)
        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * num_test + [fake_label] * num_test)
        discriminator_test_loss = discriminator.evaluate(X, y, verbose=False)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # Evaluate the testing loss of G
        noise = np.random.normal(0, 1, (2 * num_test, latent_size))
        trick = np.ones(2 * num_test)
        generator_test_loss = combined.evaluate(noise, trick, verbose=False)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # Generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} '.format('component', discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f}'
        print(ROW_FMT.format('generator (train)', train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)', test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)', train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)', test_history['discriminator'][-1]))

        # Save weights every epoch
        generator.save(
            join(outdir,'params_generator_epoch_{0:03d}.hdf5'.format(epoch)), True)
        discriminator.save(
            join(outdir,'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch)), True)

        # Generate some digits to display
        noise = np.random.normal(0, 1, (numdisplay, latent_size))

        generated_images = generator.predict(noise, verbose=0)
        pickle.dump(generated_images, open(join(outdir, 'samples_epoch_{0:03d}_generated.pkl'.format(epoch)),'wb'))

        # Dump history
        pickle.dump({'train': train_history, 'test': test_history},
                open(join(outdir,'history.pkl'), 'wb'))
