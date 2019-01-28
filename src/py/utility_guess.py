#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, LSTM, TimeDistributed
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda, Input, Dense

from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

from keras import backend as K

class ElapsedTimer:
    def __init__(self):
        self.start_time = time.time()
    def __elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        return self.__elapsed(time.time() - self.start_time)

class LaserScans:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timesteps = None
        self.cmd_vel = None
        self.scans = None
        self.scan_bound_percentage = 0

    def load(self, datafile,
             clip_scans_at=None, scan_center_range=None, scan_bound_percentage=None):
        self.clip_scans_at = clip_scans_at
        self.scan_center_range = scan_center_range
        self.scan_bound_percentage = scan_bound_percentage
        self.data = np.loadtxt(datafile).astype('float32')

        self.timesteps = self.data[:, :1]
        self.cmd_vel = self.data[:, 1:7]
        self.scans = self.data[:, 7:]

        if self.verbose:
            print("timesteps --", self.timesteps.shape)
            print("cmd_vel --", self.cmd_vel.shape)
            print("scans --", self.scans.shape, "range [", np.max(self.scans), "-", np.min(self.scans), "]")

        if not self.clip_scans_at is None:
            np.clip(self.scans, a_min=0, a_max=self.clip_scans_at, out=self.scans)

        if not self.scan_center_range is None:
            i_range = 0.5*self.scans.shape[1] - 0.5*self.scan_center_range
            i_range = i_range + 20
            self.scan_bound_percentage = 1.0 - (float(self.scan_center_range)/self.scans.shape[1])
            self.scan_bound_percentage = 0.5*self.scan_bound_percentage
            self.scans = self.scans[:, int(i_range):int(i_range) + self.scan_center_range]
        else:
            if self.scan_bound_percentage != 0:
                min_bound = int(self.scan_bound_percentage*self.scans.shape[1])
                max_bound = int(self.scans.shape[1] - self.scan_bound_percentage*self.scans.shape[1])
                self.scans = self.scans[:, min_bound:max_bound]
                if self.verbose: print("scans bounds (min, max)=", min_bound, max_bound)
        self.scans = self.scans / self.clip_scans_at    # normalization makes the vae work

    def initRand(self, rand_scans_num, scan_dim, clip_scans_at=1.0):
        self.scans = np.random.uniform(0.0, clip_scans_at, size=[rand_scans_num, scan_dim])
        self.cmd_vel = np.zeros((rand_scans_num, 6))
        self.timesteps = np.zeros((rand_scans_num, 1))

    def originalScansDim(self):
        if self.scans is None: return -1
        return self.scans.shape[1]

    def timesteps(self):
        if self.timesteps is None: return np.zeros((1, 1))
        return self.timesteps

    def cmdVel(self):
        if self.cmd_vel is None: return np.zeros((1, 1))
        return self.cmd_vel

    def getScans(self, split_at=0):
        if self.scans is None: return np.zeros((1, 1))
        if split_at == 0: return self.scans

        x_train = self.scans[:int(self.scans.shape[0]*split_at), :]
        x_test = self.scans[int(self.scans.shape[0]*split_at):, :]

        if self.verbose:
            print("scans train:", x_train.shape)
            print("scans test:", x_test.shape)

        return x_train, x_test

    def getScanSegments(self, scan, threshold):
        segments = []
        iseg = 0
        useg = bool(scan[0] > threshold)
        for d in range(scan.shape[0]):
            if useg and scan[d] < threshold:
                segments.append([iseg, d, useg])
                iseg = d
                useg = False
            if not useg and scan[d] > threshold:
                segments.append([iseg, d, useg])
                iseg = d
                useg = True
            if d == scan.shape[0] - 1: segments.append([iseg, d, useg])
        return segments

    def plotScan(self, scan, y=None):
        theta = 0.01*np.arange(0, 75, 75/self.scan_center_range)*(3/2)*np.pi - self.scan_bound_percentage*(3/2)*np.pi
        theta = theta[::-1]

        x_axis = np.arange(scan.shape[0])
        segments = self.getScanSegments(scan, 0.99)
        if self.verbose: print("Segments -- ", np.array(segments).shape, "--", segments)

        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        y_axis = scan
        if not y is None:
            y_axis = y
            plt.plot(x_axis, y_axis, color='lightgray')

        plt.plot(x_axis, scan, color='lightgray')
        for s in segments:
            if s[2]:
                col = '#ff7f0e'
                plt.plot(x_axis[s[0]:s[1]], y_axis[s[0]:s[1]], 'o', markersize=0.5, color=col)
            else:
                col = '#1f77b4'
                plt.plot(x_axis[s[0]:s[1]], scan[s[0]:s[1]], 'o', markersize=0.5, color=col)

        plt.subplot(122, projection='polar')
        plt.plot(theta, scan, color='lightgray')
        for s in segments:
            if s[2]:
                col = '#ff7f0e'
                plt.plot(theta[s[0]:s[1]], y_axis[s[0]:s[1]], 'o', markersize=0.5, color=col)
            else:
                col = '#1f77b4'
                plt.plot(theta[s[0]:s[1]], scan[s[0]:s[1]], 'o', markersize=0.5, color=col)

class VAE:
    def __init__(self, batch_size=128, latent_dim=10, intermediate_dim=128, verbose=False):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.verbose = verbose

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def buildModel(self, original_dim):
        input_shape = (original_dim, )

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        if self.verbose: encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        if self.verbose: decoder.summary()

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= original_dim

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer=Adam(lr=0.0001))
        if self.verbose: self.vae.summary()

    def fitModel(self, x, x_test=None, epochs=10, verbose=None):
        if x_test is None:
            self.vae.fit(x,
                         epochs=epochs,
                         batch_size=self.batch_size,
                         verbose=verbose)
        else:
            self.vae.fit(x,
                         epochs=epochs,
                         batch_size=self.batch_size,
                         verbose=verbose,
                         validation_data=(x_test, None))

    def predictEncoder(self, x, batch_size=None):
        if len(x.shape) == 1: x = np.array([x])
        z_mean, _, _ = self.encoder.predict(x, batch_size=batch_size)
        return z_mean

    def predictDecoder(self, z_mean):
        x_decoded = self.decoder.predict(z_mean)
        return x_decoded

class GAN:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model

    def discriminatorThin(self):
        if self.D: return self.D
        self.D = Sequential()
        depth = 16
        dropout = 0.4

        self.D.add(Conv2D(depth, 5, strides=2,
                          input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        if self.verbose: self.D.summary()
        return self.D

    def discriminator(self):
        if self.D: return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4

        self.D.add(Conv2D(depth, 5, strides=2,
                          input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        if self.verbose: self.D.summary()
        return self.D

    def generatorThin(self):
        if self.G: return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64
        dim = 16

        self.G.add(Dense(dim*depth, input_dim=2*self.latent_input_dim))  # input + noise
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 1, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        if self.verbose: self.G.summary()
        return self.G

    def generator(self):
        if self.G: return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 16

        self.G.add(Dense(dim*depth, input_dim=2*self.latent_input_dim))  # input + noise
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 1, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        if self.verbose: self.G.summary()
        return self.G

    def discriminator_model(self, model_id="default"):
        if self.DM: return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        if model_id == "thin": self.DM.add(self.discriminatorThin())
        else: self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self, model_id="default"):
        if self.AM: return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        if model_id == "thin":
            self.AM.add(self.generatorThin())
            self.AM.add(self.discriminatorThin())
        else:
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.AM

    def buildModel(self, input_shape, latent_input_dim, model_id="default"):
        self.input_shape = input_shape
        self.latent_input_dim = latent_input_dim

        self.DIS = self.discriminator_model(model_id)
        if model_id == "thin": self.GEN = self.generatorThin()
        else: self.GEN = self.generator()
        self.ADV = self.adversarial_model(model_id)

    def fitModel(self, x, x_label, train_steps=10, batch_sz=32, verbose=None):
        if verbose is None: verbose = self.verbose
        for b in range(0, x.shape[0], batch_sz):
            if b + batch_sz > x.shape[0]: continue
            for t in range(train_steps):
                noise = np.random.uniform(-1.0, 1.0, size=[batch_sz, self.latent_input_dim])
                gen_in = np.empty((batch_sz, 2*self.latent_input_dim))
                gen_in[:, ::2] = x[b:b + batch_sz]
                gen_in[:, 1::2] = noise

                real = np.zeros((batch_sz, self.input_shape[0], 1, 1))
                real[:, :, 0, 0] = x_label[b:b + batch_sz]
                fake = self.GEN.predict(gen_in)

                x_train = np.concatenate((real, fake))
                y = np.ones([x_train.shape[0], 1])
                y[batch_sz:, :] = 0

                d_loss = self.DIS.train_on_batch(x_train, y)

                y = np.ones([batch_sz, 1])
                a_loss = self.ADV.train_on_batch(gen_in, y)

                log_mesg = "%d/%d: [D loss: %f, acc: %f]" % \
                           (b + batch_sz, x.shape[0], d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                if verbose and t == train_steps - 1: print(log_mesg)

    def latentInputDim(self):
        return self.latent_input_dim

    def inputDim(self):
        return self.original_dim

    def generate(self, x):
        noise = np.random.uniform(-1.0, 1.0, size=[x.shape[0], self.latent_input_dim])
        gen_in = np.empty((x.shape[0], 2*self.latent_input_dim))
        gen_in[:, ::2] = x
        gen_in[:, 1::2] = noise
        return self.GEN.predict(gen_in)[:, :, 0, 0]

class RGAN:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model

    def discriminator(self):
        if self.D: return self.D
        self.D = Sequential()
        depth = 16
        dropout = 0.4

        self.D.add(Conv2D(depth, 5, strides=2,
                          input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        if self.verbose: self.D.summary()
        return self.D

    def generator(self):
        if self.G: return self.G
        dropout = 0.4
        depth = 64+64
        dim = 16

        self.G = Sequential()
        self.G.add(LSTM(depth, input_shape=(self.input_length_dim, 2*self.latent_input_dim),
                        return_sequences=True,
                        activation='tanh', recurrent_activation='hard_sigmoid'))
        self.G.add(LSTM(depth, return_sequences=True,
                        activation='tanh', recurrent_activation='hard_sigmoid'))
        self.G.add(Dense(dim*depth))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 1, self.input_length_dim*depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))

        if self.verbose: self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM: return self.DM
        optimizer = RMSprop(lr=0.00002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM: return self.AM
        optimizer = RMSprop(lr=0.00001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.AM

    def buildModel(self, input_shape, latent_input_dim, input_length_dim):
        self.input_shape = input_shape
        self.input_length_dim = input_length_dim
        self.latent_input_dim = latent_input_dim

        self.DIS = self.discriminator_model()
        self.GEN = self.generator()
        self.ADV = self.adversarial_model()

    def fitModel(self, x, x_label, train_steps=10, batch_sz=32, verbose=None):
        if verbose is None: verbose = self.verbose
        for b in range(0, x.shape[0], batch_sz):
            if b + batch_sz > x.shape[0]: continue
            for t in range(train_steps):
                noise = np.random.uniform(-1.0, 1.0,
                                          size=[batch_sz, self.input_length_dim, self.latent_input_dim])
                gen_in = np.empty((batch_sz, self.input_length_dim, 2*self.latent_input_dim))
                gen_in[:, :, ::2] = x[b:b + batch_sz]
                gen_in[:, :, 1::2] = noise

                real = np.zeros((batch_sz, self.input_shape[0], 1, 1))
                real[:, :, 0, 0] = x_label[b:b + batch_sz]
                fake = self.GEN.predict(gen_in)

                x_train = np.concatenate((real, fake))
                y = np.ones([x_train.shape[0], 1])
                y[batch_sz:, :] = 0

                d_loss = self.DIS.train_on_batch(x_train, y)

                y = np.ones([batch_sz, 1])
                a_loss = self.ADV.train_on_batch(gen_in, y)

                log_mesg = "%d/%d: [D loss: %f, acc: %f]" % \
                           (b + batch_sz, x.shape[0], d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                if verbose and t == train_steps - 1: print(log_mesg)

    def latentInputDim(self):
        return self.latent_input_dim

    def inputDim(self):
        return self.original_dim

    def generate(self, x):
        noise = np.random.uniform(-1.0, 1.0, size=[x.shape[0], self.input_length_dim, self.latent_input_dim])
        gen_in = np.empty((x.shape[0], self.input_length_dim, 2*self.latent_input_dim))
        gen_in[:, :, ::2] = x
        gen_in[:, :, 1::2] = noise
        return self.GEN.predict(gen_in)[:, :, 0, 0]

if __name__ == "__main__":
    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    ls = LaserScans(verbose=True)
    ls.load("../../dataset/diag_underground.txt",
             clip_scans_at=8, scan_center_range=512)

    vae = VAE()
    vae.buildModel(ls.originalScansDim())
    vae.fitModel(ls.getScans(), epochs=40, verbose=0)
    x, x_test = ls.getScans(0.9)
    vae.fitModel(x, x_test=x_test)

    batch_sz = 8
    gan_batch_sz = 8
    scan_idx = 100

    scan = x[scan_idx:(scan_idx + batch_sz*gan_batch_sz)]
    z_latent = vae.predictEncoder(scan)
    dscan = vae.predictDecoder(z_latent)
    ls.plotScan(scan[0], dscan[0])

    input_shape = (ls.originalScansDim(), 1, 1,)
    latent_dim = gan_batch_sz*(10 + 6) # latent input dim + cmdVel from vae

    gan = GAN(verbose=False)
    gan.buildModel(input_shape, latent_dim)

    cmd_vels = ls.cmdVel()[scan_idx:(scan_idx + batch_sz*gan_batch_sz)]
    x_latent = np.concatenate((z_latent, cmd_vels), axis=1)
    x_latent = np.reshape(x_latent, (batch_sz, latent_dim))

    next_scan = np.zeros((batch_sz, ls.originalScansDim()))
    for ns in range(batch_sz):
        next_scan[ns, :ls.originalScansDim()] = x[(scan_idx + gan_batch_sz + ns*gan_batch_sz + 1)]

    gan.fitModel(x_latent, next_scan, train_steps=5, batch_sz=batch_sz)

    gscan = gan.generate(x_latent)
    ls.plotScan(next_scan[0, :])
    ls.plotScan(gscan[0, :])

    plt.show()
