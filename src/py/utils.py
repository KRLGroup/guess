#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, LSTM, TimeDistributed
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda, Input, Dense
from keras.losses import mse, binary_crossentropy
from keras.models import Model, Sequential, clone_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MetricsSaver:
    def __init__(self, save_path):
        self.save_path = save_path
        self.met_dict = {}

    def add(self, mid, mrow):
        if len(mrow.shape) != 1:
            return

        if mid in self.met_dict.keys():
            if self.met_dict[mid][0].shape != mrow.shape: return
            self.met_dict[mid] = np.vstack((self.met_dict[mid], mrow))
        else:
            self.met_dict[mid] = mrow.reshape(1, mrow.shape[0])

    def save(self):
        for mid, met in self.met_dict.items():
            np.save(os.path.join(self.save_path, mid + ".npy"), met)


class ElapsedTimer:
    def __init__(self):
        self.start_time = time.time()
    def __elapsed(self, sec):
        return round(sec, 2)
    def secs(self):
        return self.__elapsed(time.time() - self.start_time)


class LaserScans:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.ts = None
        self.cmd_vel = None
        self.scans = None
        self.scan_bound = 0
        self.scan_fov = (3/2)*np.pi  # [270 deg]
        self.scan_res = 0.001389*np.pi # [0.25 deg]
        self.scan_offset = 0

    def load(self, datafile, scan_res, scan_fov,
             scan_beam_num=None, clip_scans_at=None, scan_offset=0):
        self.data = np.loadtxt(datafile).astype('float32')
        self.scan_res = scan_res
        self.scan_fov = scan_fov
        self.scan_beam_num = scan_beam_num
        self.clip_scans_at = clip_scans_at
        self.scan_offset = scan_offset

        self.ts = self.data[:, :1]
        self.cmd_vel = self.data[:, 1:7]
        self.scans = self.data[:, 7:]
        if self.verbose:
            print("-- [LasersScans] timesteps:", self.ts.shape)
            print("-- [LasersScans] cmd_vel:", self.cmd_vel.shape)
            print("-- [LasersScans] scans:", self.scans.shape,
                  "range [", np.min(self.scans), "-", np.max(self.scans), "]")

        irange = 0
        beam_num = int(self.scan_fov/self.scan_res)
        assert beam_num == self.scans.shape[1], \
            "Wrong number of scan beams " + str(beam_num) + " != " + str(self.scans.shape[1])
        if not self.scan_beam_num is None:
            if self.scan_beam_num + self.scan_offset < beam_num:
                irange = int(0.5*(beam_num - self.scan_beam_num)) + self.scan_offset
            elif self.scan_beam_num < beam_num:
                irange = int(0.5*(beam_num - self.scan_beam_num))
            self.scan_bound = (irange*self.scan_res)
        else:
            self.scan_bound = 0
            self.scan_beam_num = beam_num
        self.scans = self.scans[:, irange:irange + self.scan_beam_num]
        if self.verbose:
            r_msg = "[" + str(irange) + "-" + str(irange + self.scan_beam_num) + "]"
            print("-- [LasersScans] resized scans:", self.scans.shape, r_msg)

        if not self.clip_scans_at is None:
            np.clip(self.scans, a_min=0, a_max=self.clip_scans_at, out=self.scans)
            self.scans = self.scans / self.clip_scans_at    # normalization makes the vae work

    def init_rand(self, rand_scans_num, scan_dim, scan_res, scan_fov, clip_scans_at=5.0):
        self.scan_beam_num = scan_dim
        self.scan_res = scan_res
        self.scan_fov = scan_fov
        self.clip_scans_at = clip_scans_at
        self.scans = np.random.uniform(0, 1.0, size=[rand_scans_num, scan_dim])
        self.cmd_vel = np.zeros((rand_scans_num, 6))
        self.ts = np.zeros((rand_scans_num, 1))

    def reshape_correlated_scans(self, cmdv, ts, correlated_steps, integration_steps,
                                 theta_axis=5, normalize_factor=None):
        assert cmdv.shape[0] >= correlated_steps + integration_steps \
            and cmdv.shape[0] == ts.shape[0]

        max_prev_row = int(cmdv.shape[0]/correlated_steps)*correlated_steps
        cmdv = cmdv[:max_prev_row]
        ts = ts[:max_prev_row]

        prev_cmdv = cmdv.reshape((-1, correlated_steps, cmdv.shape[1]))
        next_cmdv = np.array([cmdv[c:c + integration_steps] for c in range(correlated_steps,
                                                                           cmdv.shape[0] - integration_steps,
                                                                           correlated_steps)], dtype=np.float32)
        rows = min(prev_cmdv.shape[0], next_cmdv.shape[0])
        prev_cmdv = prev_cmdv[:rows]
        next_cmdv = next_cmdv[:rows]
        prev_ts = 0.33*np.ones_like(prev_cmdv[..., :1])
        correlated_cmdv = np.concatenate([prev_cmdv, prev_ts], axis=-1)
        _, next_transform = self.compute_transforms(next_cmdv, theta_axis=theta_axis)

        if normalize_factor is not None:
            # translation normalization -> [-1.0, 1.0]
            next_transform[:, :2] = np.clip(next_transform[:, :2], a_min=-normalize_factor,
                                            a_max=normalize_factor)/normalize_factor
            # theta normalization -> [-1.0, 1.0]
            next_transform[:, 2] /= np.pi

        return correlated_cmdv, next_transform

    def compute_transform(self, cmdv, ts=None, theta_axis=5):
        tstep = 0.033 if ts is None else max(0.033, min(0.0, np.mean(ts[:1] - ts[:-1])))

        x, y, th = 0.0, 0.0, 0.0
        for n in range(cmdv.shape[0]):
            rk_th = th + 0.5*cmdv[n, theta_axis]*tstep  # runge-kutta integration
            x = x + cmdv[n, 0]*np.cos(rk_th)*tstep
            y = y + cmdv[n, 0]*np.sin(rk_th)*tstep
            th = th + cmdv[n, theta_axis]*tstep

        cth, sth = np.cos(th), np.sin(th)
        homogenous_matrix = np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1)), dtype=np.float32)
        transform_params = np.array([x, y, th], dtype=np.float32)
        return homogenous_matrix, transform_params

    def compute_transforms(self, cmdv, ts=None, theta_axis=5):
        transforms = [list(self.compute_transform(c, theta_axis=theta_axis)) for c in cmdv]
        homogenous_matrix = np.array([t[0] for t in transforms], dtype=np.float32)
        transform_params = np.array([t[1] for t in transforms], dtype=np.float32)
        return homogenous_matrix, transform_params

    def project_scan(self, scan, cmdv, ts):
        assert scan.shape[0] == self.scan_beam_num, "Wrong scan size."

        hm, _, _, _ = self.compute_transform(cmdv, ts)
        theta = self.scan_res*np.arange(-0.5*self.scan_beam_num, 0.5*self.scan_beam_num)
        pts = np.ones((3, self.scan_beam_num))
        pts[0] = scan*np.cos(theta)
        pts[1] = scan*np.sin(theta)
        pts = np.matmul(hm, pts)

        x2 = pts[0]*pts[0]
        y2 = pts[1]*pts[1]
        return np.sqrt(x2 + y2)

    def project_scans(self, scans, cmdv, ts):
        pscans = np.empty(scans.shape)
        for i in range(scans.shape[0]):
            pscans[i] = self.project_scan(scans[i], cmdv[i], ts[i])
        return pscans

    def scans_dim(self):
        return -1 if self.scans is None else self.scans.shape[1]

    def interpolate_scan_points(self, sp):
        # calculate polynomial
        z = np.polyfit(np.arange(sp.shape[0]), sp, deg=9)
        yp = np.poly1d(z)(np.linspace(0, sp.shape[0], sp.shape[0]))
        return yp

    def timesteps(self):
        return np.zeros((1, 1)) if self.ts is None else self.ts

    def get_cmd_vel(self):
        return np.zeros((1, 1)) if self.cmd_vel is None else self.cmd_vel

    def get_scans(self, split_at=0.):
        assert self.scans is not None, 'Empty scan array.'
        if split_at <= 0:
            return self.scans

        x_train = self.scans[:int(self.scans.shape[0]*split_at), :]
        x_test = self.scans[int(self.scans.shape[0]*split_at):, :]
        if self.verbose:
            print("-- [LasersScans] scans train:", x_train.shape)
            print("-- [LasersScans] scans test:", x_test.shape)
        return x_train, x_test

    def get_scan_segments(self, scan, threshold):
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

            if d == scan.shape[0] - 1:
                segments.append([iseg, d, useg])

        return segments

    def plot_scans(self, plot_scan_specs, title='', fig_path=""):
        assert len(plot_scan_specs) > 0, 'Empty input.'
        for spec in plot_scan_specs:
            assert spec[0].shape[0] == self.scan_beam_num, "Wrong scan size."

        x_axis = np.arange(self.scan_beam_num)
        theta = self.scan_res*np.arange(-0.5*self.scan_beam_num, 0.5*self.scan_beam_num)
        theta = theta[::-1]

        plt.figure(figsize=(5, 5))
        plt.title(title)

        ax = plt.subplot(111, projection='polar')
        ax.set_theta_offset(0.5*np.pi)
        ax.set_rlabel_position(-180)

        for spec in plot_scan_specs:
            scan, color, name = spec
            for s, segment in enumerate(self.get_scan_segments(scan, 0.99)):
                if not segment[2]:
                    segment_slice = slice(segment[0], segment[1], None)
                    plt.plot(theta[segment_slice], scan[segment_slice], 'o', markersize=0.5,
                             color=color, label=name if s == 0 else None)
        plt.legend()

        if len(fig_path) > 0:
            plt.savefig(fig_path, format='pdf')

    def plot_projections(self, scan, params, param_names=[], fig_path=""):
        assert scan.shape[0] == self.scan_beam_num, "Wrong scan size."
        assert len(param_names) == 0 or len(params) == len(param_names), "param_names must have same length of params."

        theta = self.scan_res*np.arange(-0.5*self.scan_beam_num, 0.5*self.scan_beam_num)
        pts = np.vstack([scan*np.cos(theta), scan*np.sin(theta), np.ones((self.scan_beam_num))])

        plt.figure()
        plt.axis('equal')
        plt.plot(pts[1], pts[0], label='reference')

        for p, param in enumerate(params):
            x, y, th = param[0], param[1], param[2]
            cth, sth = np.cos(th), np.sin(th)
            hm = np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1)))
            t_pts = np.matmul(hm, pts)
            plt.plot(t_pts[1], t_pts[0], label=param_names[p] if param_names else str(p))

        plt.legend()

        if fig_path != "":
            plt.savefig(fig_path, format='pdf')


class TfPredictor:
    def __init__(self, correlated_steps, input_dim, output_dim,
                 model_id="dense", batch_size=32, verbose=False):
        self.verbose = verbose
        self.correlated_steps = correlated_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.model = None
        self.model_id = model_id

    def lstm(self):
        dropout = 0.4
        depth = 64+64

        model = Sequential()
        model.add(LSTM(depth, input_shape=(self.correlated_steps, self.input_dim),
                          return_sequences=True, activation='tanh',
                          recurrent_activation='hard_sigmoid'))
        model.add(Dense(depth))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(self.output_dim, use_bias=True))
        model.add(Activation('tanh'))

        if self.verbose:
            model.summary()

        return model

    def dense(self):
        depth = 16

        model = Sequential()
        model.add(Conv1D(int(depth/4), 4, padding='same', input_shape=(self.correlated_steps, self.input_dim)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(int(depth/8), 4, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(self.output_dim, use_bias=True))
        model.add(Activation('tanh'))

        if self.verbose:
            model.summary()

        return model

    def build_model(self, lr=0.0002):
        if self.model is None:
            self.model = Sequential()

            if self.model_id == "lstm":
                self.model.add(self.lstm())
            else:
                self.model.add(self.dense())

            # optimizer = Adam(lr=lr, beta_1=0.5, decay=3e-8)
            optimizer = SGD(lr=lr)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

        return self.model

    def train(self, x, tfp_transform, epochs=10):
        ret = []
        for e in range(epochs):
            for i in range(0, x.shape[0], self.batch_size):
                met = self.model.train_on_batch(x[i:i + self.batch_size], tfp_transform[i:i + self.batch_size])
                ret.append(met)

        return np.mean(np.zeros((2,)) if len(ret) == 0 else np.array(ret), axis=0)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == "__main__":
    scan_n = 8000
    correlated_steps = 8
    batch_sz = 64
    integration_step = 10

    max_vel = 0.45
    max_dist = 0.33*integration_step*max_vel
    learning_rate = 0.002

    # diag_first_floor.txt ; diag_underground.txt ; diag_labrococo.txt
    cwd = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), "diag_underground.txt")

    ls = LaserScans(verbose=True)
    ls.load(dataset_file, scan_res=0.00653590704, scan_fov=(3/2)*np.pi, scan_beam_num=512,
            clip_scans_at=8, scan_offset=8)
    scans = ls.get_scans()[:scan_n]
    cmdvs = ls.get_cmd_vel()[:scan_n, ::5]
    timesteps = ls.timesteps()[:scan_n]

    correlated_cmdv, target_transform = ls.reshape_correlated_scans(cmdvs, timesteps,
                                                                    correlated_steps, integration_step,
                                                                    normalize_factor=max_dist, theta_axis=1)

    tfp = TfPredictor(correlated_steps=correlated_steps, input_dim=correlated_cmdv.shape[-1],
                      output_dim=3, batch_size=batch_sz, verbose=True)
    tfp.build_model(lr=learning_rate)

    tfp_transform = tfp.predict(correlated_cmdv)
    tfp_transform[:, :2] = tfp_transform[:, :2]*max_dist
    tfp_transform[:, 2] = tfp_transform[:, 2]*np.pi

    rnd_idx = int(np.random.rand() * tfp_transform.shape[0])
    ls.plot_projections(scans[rnd_idx], params=[target_transform[rnd_idx], tfp_transform[rnd_idx]],
                        param_names=['projected', 'predicted'])

    ms = MetricsSaver(save_path="/tmp/")

    nsteps = 40
    for i in range(nsteps):
        metrics = tfp.train(correlated_cmdv, target_transform, epochs=10)
        print("-- step %d: simple tfp: [loss acc]" % i, metrics)
        ms.add('projector', metrics)

    tfp_transform = tfp.predict(correlated_cmdv)
    tfp_transform[:, :2] = tfp_transform[:, :2]*max_dist
    tfp_transform[:, 2] = tfp_transform[:, 2]*np.pi

    ls.plot_projections(scans[rnd_idx], params=[target_transform[rnd_idx], tfp_transform[rnd_idx]],
                        param_names=['projected', 'predicted'])
    ms.save()
    plt.show()
