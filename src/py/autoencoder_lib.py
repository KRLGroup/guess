#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, LSTM, TimeDistributed, Conv1D, Conv1DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda, Input, Dense
from keras.models import Model, Sequential, clone_model
from keras.losses import mse, binary_crossentropy, mae
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.utils import plot_model
from keras import backend as K

import datetime

from utils import LaserScans, Landmarks

class AutoEncoder:
    def __init__(self, original_dim, variational=True, convolutional=True,
                 batch_size=128, latent_dim=10, intermediate_dim=128, verbose=False):
        self.original_dim = original_dim
        self.variational = variational
        self.convolutional = convolutional
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.verbose = verbose
        self.reshape_rows = 2
        self.encoder = None
        self.decoder = None
        # self.pencoder = None
        # self.pdecoder = None
        self.ae = None

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def __create_encoder(self, e_in, depth=32, dropout=0.4, tr=True):
        if self.convolutional:
            enc = Reshape((int(self.original_dim/self.reshape_rows), self.reshape_rows))(e_in)
            enc = Conv1D(depth, 2, activation='relu', strides=1, padding='same', trainable=tr)(enc)
            enc = LeakyReLU(alpha=0.2)(enc)
            enc = Conv1D(depth*4, 5, strides=2, padding='same', trainable=tr)(enc)
            enc = LeakyReLU(alpha=0.2)(enc)
            enc = Dropout(dropout)(enc)
            # shape info needed to build decoder model
            self.enc_shape = K.int_shape(enc)
            # Out: 1-dim probability
            enc = Flatten()(enc)

            # enc = Reshape((self.reshape_rows, int(self.original_dim/self.reshape_rows), 1,))(e_in)
            # enc = Conv2D(depth, 5, activation='relu', strides=2, padding='same', trainable=tr)(enc)
            # enc = Conv2D(depth*4, 5, strides=2, padding='same', trainable=tr)(enc)
            # enc = LeakyReLU(alpha=0.2)(enc)
            # enc = Dropout(dropout)(enc)
            # # shape info needed to build decoder model
            # self.enc_shape = K.int_shape(enc)
            # # Out: 1-dim probability
            # enc = Flatten()(enc)
        else:
            enc = e_in
            enc = Dense(self.intermediate_dim, activation='relu', trainable=tr)(enc)
            enc = Dropout(dropout)(enc)

        enc = Dense(self.intermediate_dim, activation='relu', trainable=tr)(enc)
        enc = LeakyReLU(alpha=0.2)(enc)

        if self.variational:
            self.z_mean = Dense(self.latent_dim, activation='tanh', trainable=tr, name='z_mean')(enc)
            self.z_log_var = Dense(self.latent_dim, activation='tanh', trainable=tr, name='z_log_var')(enc)
            z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
            encoder = Model(e_in, [self.z_mean, self.z_log_var, z], name='encoder')
        else:
            e_out = Dense(self.latent_dim, activation='tanh', trainable=tr)(enc)
            encoder = Model(e_in, e_out, name='encoder')
        return encoder

    def __create_decoder(self, d_in, depth=32, dropout=0.4, tr=True):
        dec = Dense(self.intermediate_dim, activation='relu', trainable=tr)(d_in)
        if self.convolutional:
            dec = Dense(self.enc_shape[1]*self.enc_shape[2],
                        activation='relu', trainable=tr)(dec)
            dec = Reshape((self.enc_shape[1], self.enc_shape[2]))(dec)

            dec = Conv1DTranspose(filters=int(depth), kernel_size=5,
                              activation='relu', strides=2, padding='same', trainable=tr)(dec)
            dec = Conv1DTranspose(filters=int(depth), kernel_size=5,
                              activation='relu', strides=2, padding='same', trainable=tr)(dec)
            dec = Dropout(dropout)(dec)
            dec = Dense(int(depth/16), activation='relu', trainable=tr)(dec)
            dec = Flatten()(dec)

            # dec = Dense(self.enc_shape[1]*self.enc_shape[2]*self.enc_shape[3],
            #             activation='relu', trainable=tr)(d_in)
            # dec = Reshape((self.enc_shape[1], self.enc_shape[2], self.enc_shape[3]))(dec)

            # dec = Conv2DTranspose(filters=int(depth/2), kernel_size=5,
            #                   activation='relu', strides=2, padding='same', trainable=tr)(dec)
            # dec = Conv2DTranspose(filters=int(depth/4), kernel_size=5,
            #                   activation='relu', strides=2, padding='same', trainable=tr)(dec)
            # dec = Dropout(dropout)(dec)
            # dec = Dense(int(depth/16), activation='relu', trainable=tr)(dec)
            # dec = Flatten()(dec)
        else:
            dec = Dense(self.intermediate_dim, activation='relu', trainable=tr)(d_in)
            dec = Dense(self.intermediate_dim, activation='relu', trainable=tr)(dec)

        d_out = Dense(self.original_dim, activation='tanh', trainable=tr)(dec)
        decoder = Model(d_in, d_out, name='decoder')
        return decoder

    def build_model(self, lr=0.01):
        if self.ae is not None:
            return self.ae

        input_shape = (self.original_dim,)
        depth = 32
        dropout = 0.2

        ## ENCODER
        e_in = Input(shape=input_shape, name='encoder_input')
        self.encoder = self.__create_encoder(e_in, depth=depth, dropout=dropout, tr=True)

        ## DECODER
        d_in = Input(shape=(self.latent_dim,), name='decoder_input')
        self.decoder = self.__create_decoder(d_in, depth=depth, dropout=dropout, tr=True)

        ## AUTOENCODER
        if self.variational:
            vae_out = self.decoder(self.encoder(e_in)[2])
            self.ae = Model(e_in, vae_out, name='vae_mlp')
            reconstruction_loss = mae(e_in, vae_out) * 10 * self.reshape_rows  # binary_crossentropy
            reconstruction_loss *= self.original_dim

            kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.square(K.exp(self.z_log_var))
            kl_loss = -0.5*K.sum(kl_loss, axis=-1)
            vae_loss = K.mean(reconstruction_loss + kl_loss)

            self.ae.add_loss(vae_loss)
            self.ae.compile(optimizer=Adam(lr=lr), metrics=['accuracy'])
        else:
            vae_out = self.decoder(self.encoder(e_in))
            self.ae = Model(e_in, vae_out, name='autoencoder')
            self.ae.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

        if self.verbose:
            self.encoder.summary()
            self.decoder.summary()
            self.ae.summary()

        return self.ae

    def train(self, x, x_test=None, epochs=10, verbose=None):
        verbose = 0 if verbose is None else verbose
        x_test = None if x_test is None else (x_test, None)

        ret = []
        if self.variational:
            met = self.ae.fit(x, x, epochs=epochs, batch_size=self.batch_size,
                              shuffle=True, validation_data=x_test, verbose=verbose)
            ret = [[l, -1] for l in met.history['loss']]
        else:
            for e in range(epochs):
                for i in range(0, x.shape[0] - self.batch_size, self.batch_size):
                    met = self.ae.train_on_batch(x[i:i + self.batch_size], x[i:i + self.batch_size])
                    ret.append(met)

        ret_avgs = np.mean(ret, axis=0)
        return np.array(ret_avgs)

    def encode(self, x, batch_size=None):
        x = np.expand_dims(x, axis=0) if len(x.shape) == 1 else x

        if self.variational:
            z_mean, _, _ = self.encoder.predict(x, batch_size=batch_size)
            return z_mean
        else:
            return self.encoder.predict(x, batch_size=batch_size)

    def decode(self, z_mean):
        return self.decoder.predict(z_mean)

if __name__ == "__main__":
    batch_sz = 1024
    landmarks_n = 20000
    learning_rate = 0.001
    latent_dim = 32
    epochs = 2500

    dataset_name = 'landmarks.txt'
    plot_indices = {
        'landmarks.txt': [i for i in range(0, landmarks_n, 1000)],
    }

    cwd = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), dataset_name)

    lms = Landmarks(verbose=True)
    lms.load(dataset_file)
    x = lms.landmarks[:landmarks_n]
    rnd_indices = np.arange(landmarks_n)
    np.random.shuffle(rnd_indices)

    ae = AutoEncoder(lms.landmarks_dim(), variational=True, convolutional=False,
                     batch_size=batch_sz, latent_dim=latent_dim, verbose=True, intermediate_dim=128)
    ae.build_model(lr=learning_rate)

    ae.train(x[rnd_indices], epochs=1)
    print('-- step 0: Fitting VAE model done.')

    decoded_x = ae.decode(ae.encode(x))
    # lms.plot([(x[5], '#e41a1c', 'scan'), (decoded_x[5], '#ff7f0e', 'decoded')])

    np.random.shuffle(rnd_indices)
    metrics = ae.train(x[rnd_indices], epochs=epochs, verbose=2)
    print('-- step 1: Fitting VAE model done.')


    save_experiment = True
    cwd = os.path.dirname(os.path.abspath(__file__))
    dtn = datetime.datetime.now()
    dt = str(dtn.month) + "-" + str(dtn.day) + "_" + str(dtn.hour) + "-" + str(dtn.minute)
    save_path_dir = os.path.join(cwd, "../../dataset/metrics/")
    save_path_dir = os.path.join(save_path_dir, "VAE" + "_" + dt) if save_experiment else ''
    os.mkdir(save_path_dir)
    save_pattern = '' if len(save_path_dir) == 0 else os.path.join(save_path_dir, '%d_sample.pdf')

    decoded_x = ae.decode(ae.encode(x))
    [lms.plot([(x[i], '#e41a1c', 'scan'), (decoded_x[i], '#ff7f0e', 'decoded')], fig_path=save_pattern if len(save_pattern) == 0 else save_pattern % i)
     for i in plot_indices[dataset_name]]
    # plt.show()


# Used for training original GUESs based on Laser Scans
# if __name__ == "__main__":
#     batch_sz = 256
#     scan_n = 5000
#     learning_rate = 0.01
#     latent_dim = 32
#     epochs = 100

#     dataset_name = 'diag_first_floor.txt'
#     plot_indices = {
#         'diag_first_floor.txt': [1000, 1500],
#         'diag_underground.txt': [1000, 6500],
#         'diag_labrococo.txt': [800, 2499],
#     }


#     cwd = os.path.dirname(os.path.abspath(__file__))
#     dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), dataset_name)

#     ls = LaserScans(verbose=True)
#     ls.load(dataset_file, scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
#             scan_beam_num=512, clip_scans_at=8, scan_offset=8)
#     x = ls.get_scans()[:scan_n]

#     rnd_indices = np.arange(scan_n)
#     np.random.shuffle(rnd_indices)

#     ae = AutoEncoder(ls.scans_dim(), variational=True, convolutional=False,
#                      batch_size=batch_sz, latent_dim=latent_dim, verbose=True)
#     ae.build_model(lr=learning_rate)

#     ae.train(x[rnd_indices], epochs=1)
#     print('-- step 0: Fitting VAE model done.')

#     # decoded_x = ae.decode(ae.encode(x))
#     # ls.plot_scans([(x[plot_idx], '#e41a1c', 'scan'), (decoded_x[plot_idx], '#ff7f0e', 'decoded')])

#     np.random.shuffle(rnd_indices)
#     metrics = ae.train(x[rnd_indices], epochs=epochs, verbose=2)
#     print('-- step 1: Fitting VAE model done.')

#     decoded_x = ae.decode(ae.encode(x))
#     [ls.plot_scans([(x[i], '#e41a1c', 'scan'), (decoded_x[i], '#ff7f0e', 'decoded')])
#      for i in plot_indices[dataset_name]]
#     plt.show()
