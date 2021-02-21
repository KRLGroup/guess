#!/usr/bin/env python
# coding: utf-8

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape, Input, concatenate, multiply
from keras.layers import Conv1D, Conv2DTranspose, UpSampling1D, UpSampling2D, LSTM
from keras.layers import LeakyReLU, Dropout, Lambda
from keras.layers import BatchNormalization, Dense
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from keras import backend as K

from utils import LaserScans
from autoencoder_lib import AutoEncoder

class GAN:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.generator_net = None
        self.discriminator = None
        self.adversarial = None

    def build_model(self, discriminator_input_shape, generator_input_shape,
                    discriminator_lr=0.0002, generator_lr=0.002, smoothing_factor=0.05,
                    noise_dim=5, noise_magnitude=2., multiply_noise=False, model_id="afmk"):
        assert noise_dim > 0, 'Noise dimension must be graeater than 1.'

        self.model_id = model_id
        self.noise_dim = noise_dim
        self.noise_magnitude = noise_magnitude
        self.smoothing_factor = smoothing_factor
        self.discriminator_input_shape = discriminator_input_shape
        self.generator_input_shape = generator_input_shape

        if not multiply_noise:
            assert self.noise_dim % self.generator_input_shape[0] == 0, \
                'Number of noise dimension must be a multiplier of %d' % self.generator_input_shape[0]

        # Build and compile the discriminator
        optimizer = Adam(lr=discriminator_lr, beta_1=0.5, decay=3e-8)
        self.discriminator = self.build_multiply_noise_discriminator() if multiply_noise else self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_multiply_noise_generator() if multiply_noise else self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.noise_dim,))
        label = Input(shape=self.generator_input_shape)
        x = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated data as input and determines validity
        # and the label of that data
        valid = self.discriminator([x, label])

        optimizer = Adam(lr=generator_lr, beta_1=0.5, decay=3e-8)
        self.adversarial = Model([noise, label], valid)
        self.adversarial.compile(loss=['binary_crossentropy'], optimizer=optimizer)

        if self.verbose:
            self.adversarial.summary()

    def build_multiply_noise_discriminator(self):
        depth = 16

        model = Sequential()
        model.add(Dense(depth, input_shape=self.discriminator_input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        if self.verbose:
            model.summary()

        x = Input(shape=self.discriminator_input_shape)
        label = Input(shape=self.generator_input_shape, dtype='float32')
        flat_label = Dense(self.discriminator_input_shape[0], trainable=False)(Flatten()(label))
        model_input = multiply([x, flat_label])
        validity = model(model_input)

        return Model([x, label], validity)

    def build_discriminator(self):
        depth = 16
        discriminator_input_shape = (self.discriminator_input_shape[0]
                                     + np.prod(self.generator_input_shape),)

        model = Sequential()
        model.add(Dense(depth, input_shape=discriminator_input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        if self.verbose:
            model.summary()

        x = Input(shape=self.discriminator_input_shape)
        label = Input(shape=self.generator_input_shape, dtype='float32')
        label_flat = Flatten()(label)
        model_input = concatenate([x, label_flat])

        validity = model(model_input)

        return Model([x, label], validity)

    def build_multiply_noise_generator(self):
        depth = 64
        model = Sequential()

        if self.model_id == 'conv':
            model.add(Conv1D(int(depth/2), 4, padding='same', input_shape=self.generator_input_shape))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Flatten())
            model.add(Dense(int(depth)))
        else:
            model.add(Flatten(input_shape=self.generator_input_shape))
            model.add(Dense(int(depth)))

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.discriminator_input_shape), activation='tanh'))

        if self.verbose:
            model.summary()

        noise_input = Input(shape=(self.noise_dim,))
        label_input = Input(shape=self.generator_input_shape)

        noise_embedding = Flatten()(Embedding(self.noise_dim, int(np.prod(self.generator_input_shape)/self.noise_dim))(noise_input))
        noise_dense = Dense(np.prod(self.generator_input_shape), trainable=False)(noise_embedding)
        noise = Reshape(self.generator_input_shape)(noise_dense)
        model_input = multiply([noise, label_input])

        x = model(model_input)

        return Model([noise_input, label_input], x)

    def build_generator(self):
        depth = 64
        model = Sequential()
        noise_dim = int(self.noise_dim/self.generator_input_shape[0])
        generator_input_shape = (self.generator_input_shape[0], self.generator_input_shape[1] + noise_dim,)

        if self.model_id == 'conv':
            model.add(Conv1D(int(depth/2), 4, padding='same', input_shape=generator_input_shape))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Flatten())
            model.add(Dense(int(depth)))
        else:
            model.add(Flatten(input_shape=generator_input_shape))
            model.add(Dense(int(depth)))

        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.discriminator_input_shape), activation='tanh'))

        if self.verbose:
            model.summary()

        noise_input = Input(shape=(self.noise_dim,))
        label_input = Input(shape=self.generator_input_shape)

        noise = Reshape((self.generator_input_shape[0], noise_dim))(noise_input)
        model_input = concatenate([noise, label_input])
        x = model(model_input)

        return Model([noise_input, label_input], x)

        # elif self.model_id == "lstm":
        #     self.generator_net.add(LSTM(depth, input_shape=self.generator_input_shape,
        #                                 return_sequences=True, activation='tanh',
        #                                 recurrent_activation='hard_sigmoid'))
        #     self.generator_net.add(LSTM(int(depth/2), return_sequences=True,
        #                                 activation='tanh', recurrent_activation='hard_sigmoid'))
        #     self.generator_net.add(Dense(depth))
        #     self.generator_net.add(BatchNormalization(momentum=0.9))
        #     self.generator_net.add(LeakyReLU(alpha=0.2))
        #     self.generator_net.add(Dropout(dropout))
        #     self.generator_net.add(UpSampling1D(4))
        #     self.generator_net.add(Conv1D(int(depth/4), 5, padding='same'))
        #     self.generator_net.add(BatchNormalization(momentum=0.9))
        #     self.generator_net.add(LeakyReLU(alpha=0.2))
        #     self.generator_net.add(Flatten())
        #     self.generator_net.add(Dense(self.discriminator_input_shape[0]))
        #     self.generator_net.add(Activation('tanh'))

    def set_trainable(self, net, tr=False):
        net.trainable = tr
        for l in net.layers:
            l.trainable = tr
        return net

    def train(self, x, y, batch_sz=32, train_steps=10, verbose=None):
        assert x.shape[0] == y.shape[0], "Wrong input size."
        verbose = self.verbose if verbose is None else verbose
        dataset_n_samples = int(x.shape[0]/batch_sz)*batch_sz
        x = x[:dataset_n_samples]
        y = y[:dataset_n_samples]

        target_label = np.ones((batch_sz, 1))
        real_label = 1. - np.random.uniform(0.0, self.smoothing_factor, size=((dataset_n_samples, 1)))
        fake_label = np.random.uniform(0.0, self.smoothing_factor, size=((dataset_n_samples, 1)))

        metrics = []
        for t in range(train_steps):
            d_loss, d_acc, a_loss = 0., 0., 0.

            for b in range(0, dataset_n_samples, batch_sz):
                batch_slice = slice(b, b + batch_sz, None)
                noise = (np.random.rand(batch_sz, self.noise_dim)*self.noise_magnitude) - 0.5*self.noise_magnitude
                fake = self.generator.predict([noise, x[batch_slice]])
                real = y[batch_slice]

                # fit discriminator
                d_loss_real = self.discriminator.train_on_batch([real, x[batch_slice]], real_label[batch_slice])
                d_loss_fake = self.discriminator.train_on_batch([fake, x[batch_slice]], fake_label[batch_slice])
                d_loss, d_acc = (0.5*np.add(d_loss_real, d_loss_fake)).tolist()

                # fit generator
                a_loss = self.adversarial.train_on_batch([noise, x[batch_slice]], target_label)

            metrics.append([d_loss, d_acc, a_loss])
            if verbose:
                log_msg = "-- %d/%d: [D loss: %f, acc: %f] - [A loss: %f]" % (t + 1, train_steps,
                                                                              d_loss, d_acc, a_loss)
                print(log_msg)
                sys.stdout.write("\033[F\033[K")

        if verbose:
            print(log_msg)

        return np.array(metrics)

    def generate(self, x):
        noise = (np.random.rand(1, self.noise_dim)*self.noise_magnitude) - 0.5*self.noise_magnitude
        sample = self.generator.predict([noise, np.expand_dims(x, axis=0) if len(x.shape) == 2 else x])
        return sample[0] if len(x.shape) == 2 else sample


if __name__ == "__main__":
    # params
    scan_n = 10000
    dataset_name = 'diag_first_floor.txt'
    plot_indices = {
        'diag_first_floor.txt': [1000, 1500],
        'diag_underground.txt': [1000, 6500],
        'diag_labrococo.txt': [800],
    }
    scan_to_predict_idx = plot_indices[dataset_name][0]

    scan_beam_num = 512
    latent_dim = 32
    correlated_sequence_step = 8
    prediction_step = 8

    ae_batch_sz = 256
    ae_epochs = 100

    gan_noise_dim = 1
    gan_batch_sz = 32
    gan_train_steps = 500
    gan_predict_encodings = False

    cwd = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), dataset_name)

    # ---- laser-scans
    ls = LaserScans(verbose=True)
    ls.load(dataset_file, scan_res=0.00653590704, scan_fov=(3/2)*np.pi, scan_beam_num=scan_beam_num,
            clip_scans_at=8, scan_offset=8)
    scans = ls.get_scans()[:scan_n]
    cmdv = ls.get_cmd_vel()[:scan_n, ::5]
    cmdv_dim = cmdv.shape[-1]

    correlated_steps = slice((scan_to_predict_idx - (prediction_step + correlated_sequence_step)),
                             (scan_to_predict_idx - prediction_step), None)
    scan_to_predict = scans[scan_to_predict_idx]
    correlated_scan_sequence = scans[correlated_steps]
    correlated_cmdv_sequence = cmdv[correlated_steps]

    rnd_indices = np.arange(scans.shape[0])
    np.random.shuffle(rnd_indices)

    # ---- autoencoder
    ae = AutoEncoder(ls.scans_dim(), variational=True, convolutional=False,
                     batch_size=ae_batch_sz, latent_dim=latent_dim, verbose=False)
    ae.build_model(lr=0.01)
    print('-- Fitting VAE model done.')

    ae.train(scans[rnd_indices], epochs=ae_epochs)
    encoded_scans = ae.encode(scans)
    decoded_scans = ae.decode(encoded_scans)

    rnd_idx = int(np.random.rand() * scan_n)
    # ls.plot_scans(scans[rnd_idx], decoded_scans[rnd_idx])
    # ls.plot_scans(scan_to_predict, decoded_scans[scan_to_predict_idx])
    # plt.show()

    correlated_latent = np.concatenate((ae.encode(correlated_scan_sequence),
                                        correlated_cmdv_sequence), axis=-1)

    # ---- gan - generating latent spaces
    gan = GAN(verbose=True)
    gan.build_model(discriminator_input_shape=(latent_dim if gan_predict_encodings else scan_beam_num,),
                    generator_input_shape=(correlated_sequence_step, latent_dim + cmdv_dim,),
                    discriminator_lr=1e-2, generator_lr=1e-4, smoothing_factor=0.,
                    noise_dim=gan_noise_dim, noise_magnitude=1., model_id="conv")

    latent = np.concatenate([encoded_scans, cmdv], axis=-1)
    gan_x = latent.reshape((-1, correlated_sequence_step, latent_dim + cmdv_dim))

    if gan_predict_encodings:
        gan_y = encoded_scans[(correlated_sequence_step + prediction_step)::correlated_sequence_step]
    else:
        gan_y = scans[(correlated_sequence_step + prediction_step)::correlated_sequence_step]
    # [plt.plot(gan_y[int(np.random.rand() * gan_y.shape[0])]) for _ in range(10)]

    dataset_dim = min(gan_x.shape[0], gan_y.shape[0])
    gan_x = gan_x[:dataset_dim]
    gan_y = gan_y[:dataset_dim]

    rnd_indices = np.arange(dataset_dim)
    np.random.shuffle(rnd_indices)
    metrics = gan.train(gan_x[rnd_indices], gan_y[rnd_indices], train_steps=gan_train_steps,
                        batch_sz=gan_batch_sz, verbose=True)

    if gan_predict_encodings:
        gen_out = gan.generate(correlated_latent)
        gen_scan = ae.decode(gen_out.reshape((1, latent_dim)))[0]
    else:
        gen_out = gan.generate(correlated_latent)
        gen_scan = gen_out

    plt.title('Metrics')
    color_dict = {
        "red" : np.array([251, 180, 174])/255.0,
        "blue" : np.array([179, 205, 227])/255.0,
    }
    markers = ['^', 'o', 's', '*', '+']

    plt.plot(metrics[:, 0], label='Discriminator', lw=1.2, color=0.9*color_dict['red'],
             marker=markers[0], markersize=7, markevery=50)
    plt.plot(metrics[:, 2], label='Generator', lw=1.2, color=0.9*color_dict['blue'],
             marker=markers[3], markersize=7, markevery=50)
    plt.grid(color=np.array([210, 210, 210])/255.0, linestyle='--', linewidth=1)
    plt.legend()

    np.save('/home/sapienzbot/Desktop/rss_guess_res/generation-loss.npy', np.concatenate([metrics[:, 0], metrics[:, 2]], axis=-1))

    plt.figure()
    plt.title('Latent %d' % scan_to_predict_idx)
    plt.plot(gen_out, label='generated')
    plt.plot(encoded_scans[scan_to_predict_idx] if gan_predict_encodings else scans[scan_to_predict_idx], label='target')
    plt.legend()

    [ls.plot_scans([(scans[i], '#e41a1c', 'scan'),
                    (decoded_scans[i], '#ff7f0e', 'decoded'),
                    (gen_scan, '#1f77b4', 'generated')], title='scans')
     for i in plot_indices[dataset_name]]

    plt.show()


    # ---- gan - generating scans

    # gan = GAN(verbose=True)
    # gan.build_model(discriminator_input_shape=(ls.scans_dim(),),
    #                generator_input_shape=(correlated_sequence_step, latent_dim + cmdv_dim),
    #                smoothing_factor=0.03, noise_dim=1, model_id="afmk")

    # latent = np.concatenate([encoded_scans, cmdv], axis=-1)
    # gan_x = latent.reshape((-1, correlated_sequence_step, latent_dim + cmdv_dim))
    # gan_y = scans[(correlated_sequence_step + prediction_step)::correlated_sequence_step]
    # dataset_dim = min(gan_x.shape[0], gan_y.shape[0])
    # gan_x = gan_x[:dataset_dim]
    # gan_y = 2*gan_y[:dataset_dim] - 1.0

    # rnd_indices = np.arange(dataset_dim)
    # np.random.shuffle(rnd_indices)
    # gan.train(gan_x[rnd_indices], gan_y[rnd_indices], train_steps=20, batch_sz=gan_batch_sz, verbose=True)

    # # for i in range(1, 8):
    # #     ls.plot_scans(correlated_scan_sequence[0], correlated_scan_sequence[i])
    # #     plt.show()

    # ls.plot_scans(correlated_scan_sequence[0], scan_to_predict)
    # ls.plot_scans(0.5*(gan.generate(correlated_latent) + 1.0), scan_to_predict)

    # del gan
