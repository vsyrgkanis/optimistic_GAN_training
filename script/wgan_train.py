#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
from six.moves import range
import numpy as np, sys, argparse, cPickle as pickle
from os.path import join, exists, abspath, dirname
from os import system, makedirs

from PIL import Image
from functools import partial
from keras.datasets import mnist
import keras.backend as K
from keras.optimizers import Adam, RMSprop, SGD, Adagrad
from keras.layers import Input
from keras.models import Model
from keras.utils.generic_utils import Progbar
import theano.tensor as T
import utils as utils
from optimizer import *
import  models

K.set_image_data_format('channels_first')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest='datadir')
    parser.add_argument("-o", dest='outdir')
    parser.add_argument("-l", dest='seqlen', type=int, default=6)
    parser.add_argument("-c", dest='nchannel', type=int, default=4)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr", dest='optimizer_lr', type=float, default=5e-3)
    parser.add_argument("-v", dest='version', type=int)
    parser.add_argument("-s", dest='schedule', default='None')
    parser.add_argument("--g_interval", dest='train_G_interval', type=int, default=5)
    parser.add_argument("-t", dest='network_type', default='wgan')
    parser.add_argument("-b", dest='batch_size', type=int, default=512)
    parser.add_argument("-e", dest='epoches', type=int, default=100)
    parser.add_argument("-p", dest='gradient_penalty', type=float, default=0.01)
    parser.add_argument("--lt", dest='latent_size', default=50, type=int)
    parser.add_argument("-n", dest='noise_distr', default='normal', type=str)
    parser.add_argument("--momentum", default=0, type=float)
    parser.add_argument("--nesterov", action='store_true')
    parser.add_argument("--ndisplay", default=100000, type=int)
    parser.add_argument("--normalized", action='store_true')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    noise_distr_map = {'normal':(np.random.normal, 0, 1), 'uniform':(np.random.uniform, 1, 2)}
    noise_distr = noise_distr_map[args.noise_distr][0]
    noise_distr_param = noise_distr_map[args.noise_distr][1:]

    if args.schedule not in ['None', 'adagrad']:
        raise ValueError('args.schedule {} not recognized'.format(args.schedule))
    schedule = args.schedule if args.schedule != 'None' else None

    # Copy over arguments and model files
    outdir = args.outdir
    if exists(outdir):
        system('rm -r ' + outdir)
    makedirs(outdir)
    system('cp ' + __file__ + ' '+ outdir)
    system('cp ' + 'models.py' + ' '+ outdir)
    with open(join(outdir, 'commandline_args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

    # Parameters that matter less
    epochs = args.epoches
    batch_size = args.batch_size
    latent_size = args.latent_size
    numdisplay = args.ndisplay
    GRADIENT_PENALTY_WEIGHT = args.gradient_penalty

    # Optimizer
    if args.optimizer == 'SGD':
        d_optim = Adagrad(lr=args.optimizer_lr) if schedule == 'adagrad' else SGD(lr=args.optimizer_lr, momentum=args.momentum, nesterov=args.nesterov)
        g_optim = Adagrad(lr=args.optimizer_lr) if schedule == 'adagrad' else SGD(lr=args.optimizer_lr, momentum=args.momentum, nesterov=args.nesterov)
    else:
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

    # Load the data
    (X_train, y_train), (X_test, y_test) = utils.load_data(join(args.datadir, 'embed/CV0'))
    if not args.normalized:
        print('Now normalize the input data')
        X_train = (X_train.astype(np.float32) - 0.5) / 0.5
    	X_test = (X_test.astype(np.float32) - 0.5) / 0.5
    num_train, num_test = X_train.shape[0], X_test.shape[0]

    # build the discriminator
    discriminator = models.build_discriminator(args.seqlen, args.nchannel, output_activation=output_activation)

    # build the generator
    generator = models.build_generator(latent_size, args.seqlen, args.nchannel)

    # we only want to be able to train generation for the combined model
    latent = Input(shape=(latent_size, ))
    utils.set_trainability(discriminator, False)
    fake = generator(latent)
    fake = discriminator(fake)
    combined = Model(latent, fake)
    combined.compile(optimizer=g_optim,
                     loss=loss_func)

    # The actual discriminator model
    utils.set_trainability(discriminator, True)
    real_samples = Input(shape=X_train.shape[1:])
    generated_samples_for_discriminator = Input(shape=X_train.shape[1:])
    discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    averaged_samples = utils.RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
    averaged_samples_out = discriminator(averaged_samples)
    partial_gp_loss = partial(utils.gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error
    discriminator_model = Model(inputs=[real_samples,
                                        generated_samples_for_discriminator],
                                outputs=[discriminator_output_from_real_samples,
                                         discriminator_output_from_generator,
                                         averaged_samples_out])

    discriminator_model.compile(optimizer=d_optim,
                          loss=[loss_func, loss_func, partial_gp_loss])


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
            utils.set_trainability(discriminator_model, True)
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            noise = noise_distr(noise_distr_param[0], noise_distr_param[1], (batch_size, latent_size))
            generated_images = generator.predict(noise, verbose=0)
            pos_y_batch = np.array([1] * batch_size)
            neg_y_batch = np.asarray([fake_label] * batch_size)
            dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
            epoch_disc_loss.append(discriminator_model.train_on_batch([image_batch, generated_images], [pos_y_batch, neg_y_batch, dummy_y]))

            if (1 + index) % args.train_G_interval == 0:
                # Train G
                utils.set_trainability(discriminator_model, False)
                noise = noise_distr(noise_distr_param[0], noise_distr_param[1], (2 * batch_size, latent_size))
                trick =  np.ones(2 * batch_size)
                epoch_gen_loss.append(combined.train_on_batch(noise, trick))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # Evaluate the testing loss of D
        noise = noise_distr(noise_distr_param[0], noise_distr_param[1], (num_test, latent_size))
        generated_images = generator.predict(noise, verbose=False)
        pos_y = np.array([1] * num_test)
        neg_y = np.asarray([fake_label] * num_test)
        d_y = np.zeros((num_test, 1), dtype=np.float32)
        discriminator_test_loss = discriminator_model.evaluate([X_test, generated_images], [pos_y, neg_y, d_y], verbose=False)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # Evaluate the testing loss of G
        noise = noise_distr(noise_distr_param[0], noise_distr_param[1], (2 * num_test, latent_size))
        trick = np.ones(2 * num_test)
        generator_test_loss = combined.evaluate(noise, trick, verbose=False)
        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # Generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)
        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} '.format('component', discriminator_model.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f}'
        print(ROW_FMT.format('generator (train)', train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)', test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)', train_history['discriminator'][-1][0]))
        print(ROW_FMT.format('discriminator (test)', test_history['discriminator'][-1][0]))

        # Save weights every epoch
        generator.save(
            join(outdir,'params_generator_epoch_{0:03d}.hdf5'.format(epoch)), True)
        discriminator.save(
            join(outdir,'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch)), True)

        # Generate some digits to display
        noise = noise_distr(noise_distr_param[0], noise_distr_param[1], (numdisplay, latent_size))

        generated_images = generator.predict(noise, verbose=0)
        pickle.dump(generated_images, open(join(outdir, 'samples_epoch_{0:03d}_generated.pkl'.format(epoch)),'wb'))

        # Dump history
        pickle.dump({'train': train_history, 'test': test_history},
                open(join(outdir,'history.pkl'), 'wb'))
