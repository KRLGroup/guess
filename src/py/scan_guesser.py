#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import sys
import numpy as np
from threading import Thread, Lock
from utils import LaserScans, MetricsSaver, ElapsedTimer, TfPredictor
from autoencoder_lib import AutoEncoder
from gan_lib import GAN
import matplotlib.pyplot as plt

class ScanGuesser:
    def __init__(self,
                 original_scan_dim, net_model="conv",
                 scan_seq_sz=8, clip_scans_at=8.0,
                 scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                 gen_step=1, max_dist_projector=1.0,
                 ae_fit=True, proj_fit=True, gan_fit=True,
                 # autoencoder configs
                 ae_variational=True, ae_convolutional=False,
                 ae_batch_sz=64, ae_latent_dim=10, ae_intermediate_dim=128, ae_epochs=20,
                 # gan configs
                 gan_batch_sz=32, gan_train_steps=5, gan_noise_dim=0,
                 start_update_thr=False, verbose=False, run_id=None, metrics_save_rate=100):
        self.start_update_thr = start_update_thr
        self.verbose = verbose
        self.original_scan_dim = original_scan_dim
        self.scan_resolution = scan_res
        self.scan_fov = scan_fov
        self.net_model = net_model
        self.scan_seq_sz = scan_seq_sz
        self.clip_scans_at = clip_scans_at
        self.gen_step = gen_step
        self.max_dist_projector = max_dist_projector
        self.ae_fit = ae_fit
        self.proj_fit = proj_fit
        self.gan_fit = gan_fit
        self.interpolate_scans_pts = False
        self.ae_epochs = ae_epochs
        self.gan_batch_sz = gan_batch_sz
        self.gan_train_steps = gan_train_steps

        self.gan_input_shape = (self.scan_seq_sz, ae_latent_dim + 6)
        self.online_scans = None
        self.online_cmdv = None
        self.online_ts = None
        self.sim_step = 0
        self.metrics_step = 0
        self.metrics_save_rate = metrics_save_rate  # save every steps
        if not run_id is None:
            dtn = datetime.datetime.now()
            dt = str(dtn.month) + "-" + str(dtn.day) + "_" + str(dtn.hour) + "-" + str(dtn.minute)
            metrics_base_path, _ = os.path.split(os.path.realpath(__file__))
            metrics_base_path = metrics_base_path + "/../../dataset/metrics/" + run_id + "_" + dt + "_"
            self.ms = MetricsSaver(metrics_base_path)
        else: self.ms = None

        self.ls = LaserScans(verbose=verbose)
        self.projector = TfPredictor(scan_seq_sz, 7, 3,
                                     batch_size=self.gan_batch_sz, verbose=verbose)
        self.ae = AutoEncoder(self.original_scan_dim,
                              variational=ae_variational, convolutional=ae_convolutional,
                              latent_dim=ae_latent_dim, intermediate_dim=ae_intermediate_dim,
                              batch_size=ae_batch_sz, verbose=verbose)
        self.gan = GAN(verbose=self.verbose)

        # building models
        self.projector.buildModel()
        self.ae.buildModel()
        self.gan.buildModel((self.original_scan_dim,), self.gan_input_shape,
                            smoothing_factor=0.1, noise_dim=gan_noise_dim, model_id=self.net_model)

        self.update_mtx = Lock()
        self.updating_model = False
        self.thr_scans = None
        self.thr_cmdv = None
        self.thr_ts = None
        if self.start_update_thr: self.thr = Thread(target=self.__updateThr)

    def __updateThr(self):
        while True:
            scans = None
            cmd_vel = None
            ts = None
            self.update_mtx.acquire()
            scans = self.thr_scans
            cmd_vel = self.thr_cmdv
            ts = self.thr_ts
            self.thr_scans = None
            self.thr_cmdv = None
            self.thr_ts = None
            self.update_mtx.release()

            if not scans is None and not cmd_vel is None and not ts is None:
                self.update_mtx.acquire()
                self.updating_model = True
                self.update_mtx.release()

                self.__fitModel(scans, cmd_vel, ts, verbose=False)

                self.update_mtx.acquire()
                self.updating_model = False
                self.update_mtx.release()
        print("-- Terminating update thread.")

    def __reshapeGanInput(self, scans, cmd_vel, ts, ae_encoded):
        x_latent = np.concatenate((ae_encoded, cmd_vel), axis=1)
        n_rows = int((ae_encoded.shape[0] - self.scan_seq_sz - self.gen_step)/self.scan_seq_sz) + 1
        x_latent = x_latent[:n_rows*self.scan_seq_sz].\
                   reshape((n_rows, self.scan_seq_sz, self.gan_input_shape[1]))
        next_scan, pparams, hp = self.ls.reshapeInSequences(scans, cmd_vel, ts,
                                                            self.scan_seq_sz, self.gen_step,
                                                            normalize=self.max_dist_projector)
        return x_latent, next_scan, pparams, hp

    def __updateAE(self, scans, verbose=None):
        if verbose is None: verbose = self.verbose
        v = 1 if verbose else 0
        if self.ae_fit: return self.ae.fitModel(scans, epochs=self.ae_epochs, verbose=v)
        else: return np.zeros((2,))

    def __updateGan(self, scans, cmd_vel, ts, verbose=False):
        latent = self.encodeScan(scans)
        x_latent, next_scan, pp, hp = self.__reshapeGanInput(scans, cmd_vel, ts, latent)
        p_metrics, g_metrics = np.zeros((2,)), np.zeros((4,))
        if self.proj_fit:
            p_metrics = self.projector.fitModel(pp, hp, epochs=40)
        if self.gan_fit:
            next_scan = 2.0*next_scan - 1.0  # tanh normalize
            g_metrics = self.gan.fitModel(x_latent, next_scan,
                                          train_steps=self.gan_train_steps,
                                          batch_sz=self.gan_batch_sz, verbose=verbose)[-1]
        return np.concatenate((p_metrics, g_metrics))

    def __fitModel(self, scans, cmd_vel, ts, verbose=False):
        print("-- Update model... ", end='')
        timer = ElapsedTimer()
        ae_metrics = self.__updateAE(scans)
        gan_metrics = self.__updateGan(scans, cmd_vel, ts)
        elapsed_secs = timer.secs()
        print("done (\033[1;32m" + str(elapsed_secs) + "s\033[0m).")
        print("  -- AE loss:", ae_metrics[0], "- acc:", ae_metrics[1])
        print("  -- Proj loss:", gan_metrics[0], "- acc:", gan_metrics[1])
        print("  -- GAN d-loss:", gan_metrics[2], "- d-acc:", gan_metrics[3], end='')
        print(" - a-loss:", gan_metrics[4], "- a-acc:", gan_metrics[5])
        if not self.ms is None:
            self.ms.add("ae_mets", ae_metrics)
            self.ms.add("gan-tf_mets", gan_metrics)
            self.ms.add("update_time", np.array([elapsed_secs]))

        if self.metrics_step + 1 == self.metrics_save_rate:
            if not self.ms is None: self.ms.save()
            self.metrics_step = 0
        else:
            self.metrics_step += 1
            if not self.verbose and elapsed_secs < 5.0:
                sys.stdout.write("\033[F\033[K")
                sys.stdout.write("\033[F\033[K")
                sys.stdout.write("\033[F\033[K")
                sys.stdout.write("\033[F\033[K")


    def init(self, raw_scans_file, init_models=False, init_scan_batch_num=None, scan_offset=0):
        print("- Initialize:")
        init_scan_num = None
        if raw_scans_file is None:
            print("-- Init random scans... ", end='')
            if init_scan_batch_num is None: init_scan_batch_num = 1
            init_scan_num = self.gan_batch_sz*self.scan_seq_sz*init_scan_batch_num + self.gen_step
            self.ls.initRand(init_scan_num,
                             self.original_scan_dim, self.scan_resolution, self.scan_fov,
                             clip_scans_at=self.clip_scans_at)
        else:
            print("-- Init scans from dataset... ", end='')
            self.ls.load(raw_scans_file, scan_res=self.scan_resolution, scan_fov=self.scan_fov,
                         scan_beam_num=self.original_scan_dim,
                         clip_scans_at=self.clip_scans_at, scan_offset=scan_offset)
        print("done.")

        if init_models:
            if init_scan_batch_num is None:
                scans = self.ls.getScans()
                cmd_vel = self.ls.cmdVel()
                ts = self.ls.timesteps()
            else:
                scans = self.ls.getScans()[:init_scan_num]
                cmd_vel = self.ls.cmdVel()[:init_scan_num]
                ts = self.ls.timesteps()[:init_scan_num]

            self.__fitModel(scans, cmd_vel, ts, verbose=False)
            if self.start_update_thr:
                print("-- Init update thread... ", end='')
                self.thr.start()
                print("done.\n")

    def addScans(self, scans, cmd_vel, ts):
        if scans.shape[0] < self.scan_seq_sz: return False
        if self.online_scans is None:
            self.online_scans = scans
            self.online_cmdv = cmd_vel
            self.online_ts = ts
        else:
            self.online_scans = np.concatenate((self.online_scans, scans))
            self.online_cmdv = np.concatenate((self.online_cmdv, cmd_vel))
            self.online_ts = np.concatenate((self.online_ts, ts))

        min_scan_num = 2*(self.gan_batch_sz*self.scan_seq_sz) + self.gen_step
        if self.online_scans.shape[0] < min_scan_num: return False
        # rnd_idx = np.random.randint(self.online_scans.shape[0] - min_scan_num + 1)
        rnd_idx = self.online_scans.shape[0] - min_scan_num

        if self.start_update_thr:
            self.update_mtx.acquire()
            if not self.updating_model:
                self.thr_scans = self.online_scans[rnd_idx:rnd_idx + min_scan_num]
                self.thr_cmdv = self.online_cmdv[rnd_idx:rnd_idx + min_scan_num]
                self.thr_ts = self.online_ts[rnd_idx:rnd_idx + min_scan_num]
            self.update_mtx.release()
        else:
            self.__fitModel(self.online_scans[rnd_idx:rnd_idx + min_scan_num],
                            self.online_cmdv[rnd_idx:rnd_idx + min_scan_num],
                            self.online_ts[rnd_idx:rnd_idx + min_scan_num],
                            verbose=False)
        return True

    def addRawScans(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.scan_seq_sz: return False
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans/self.clip_scans_at
        return self.addScans(scans, cmd_vel, ts)

    def encodeScan(self, scans):
        return self.ae.encode(scans)

    def decodeScan(self, scans_latent, interpolate=False):
        decoded = self.ae.decode(scans_latent)
        if interpolate:
            for d in range(decoded.shape[0]):
                decoded[d] = self.ls.interpolateScanPoints(decoded[d])
        decoded[decoded > 0.9] = 0.0
        return decoded

    def generateScan(self, scans, cmd_vel, ts):
        if self.verbose: timer = ElapsedTimer()
        if scans.shape[0] < self.scan_seq_sz: return None

        ae_encoded = self.encodeScan(scans)
        latent = np.concatenate((ae_encoded, cmd_vel), axis=1)
        n_rows = int(ae_encoded.shape[0]/self.scan_seq_sz)
        x_latent = latent[:ae_encoded.shape[0]].reshape((n_rows,
                                                         self.scan_seq_sz, self.gan_input_shape[1]))
        gen = self.gan.generate(x_latent)
        gen = 0.5*(gen + 1.0)  # tanh denormalize
        if self.interpolate_scans_pts:
            gen = self.ls.interpolateScanPoints(gen)
        gen[gen > 0.9] = 0.0

        ts = ts.reshape((1, ts.shape[0], 1,))
        cmd_vel = cmd_vel.reshape((1, cmd_vel.shape[0], cmd_vel.shape[1],))
        pparams = np.concatenate((cmd_vel, ts), axis=2)
        hp = self.projector.predict(pparams, denormalize=self.max_dist_projector)[0]

        if self.verbose: print("-- Prediction in", timer.secs())
        return gen, self.decodeScan(ae_encoded, interpolate=self.interpolate_scans_pts), hp

    def generateRawScan(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.scan_seq_sz: return None
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans/self.clip_scans_at
        gscan, vscan, hp = self.generateScan(scans, cmd_vel, ts)
        return gscan*self.clip_scans_at, vscan*self.clip_scans_at, hp

    def getScans(self):
        if not self.online_scans is None and self.online_scans.shape[0] > self.scan_seq_sz:
            return self.online_scans
        else: return self.ls.getScans()

    def cmdVel(self):
        if not self.online_cmdv is None and self.online_cmdv.shape[0] > self.scan_seq_sz:
            return self.online_cmdv
        else: return self.ls.cmdVel()

    def timesteps(self):
        if not self.online_ts is None and self.online_ts.shape[0] > self.scan_seq_sz:
            return self.online_ts
        else: return self.ls.timesteps()

    def plotScan(self, scan, decoded_scan=None):
        if decoded_scan is None: self.ls.plotScan(scan)
        else: self.ls.plotScan(scan, decoded_scan)

    def plotProjection(self, scan, hm_params=None, gen_params=None):
        self.ls.plotProjection(scan, params0=hm_params, params1=gen_params)

    def simStep(self):
        # print("--- SIMULATE step:",
        #       int(self.sim_step/(self.scan_seq_sz*self.gan_batch_sz)) + 1,
        #       "; #samples:", self.sim_step)
        scans_num = self.scan_seq_sz*self.gan_batch_sz + self.gen_step
        scans = self.ls.getScans()[self.sim_step:self.sim_step + scans_num]
        cmd_vel = self.ls.cmdVel()[self.sim_step:self.sim_step + scans_num]
        ts = self.ls.timesteps()[self.sim_step:self.sim_step + scans_num]

        self.sim_step = self.sim_step + self.scan_seq_sz*self.gan_batch_sz
        return self.addScans(scans, cmd_vel, ts)

if __name__ == "__main__":
    print("ScanGuesser test-main")
    scan_seq_size = 8
    scan_generation_step = 5
    guesser = ScanGuesser(512, # number of scan beams considered
                          net_model="afmk",  # conv; lstm
                          scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                          scan_seq_sz=scan_seq_size,  # sequence of scans as input
                          gen_step=scan_generation_step, # \# of 'scan steps' to look ahead
                          ae_fit=True, proj_fit=True, gan_fit=True,
                          # autoencoder configs
                          ae_epochs=10, ae_variational=True, ae_convolutional=False,
                          ae_latent_dim=10,
                          # gan configs
                          gan_batch_sz=32, gan_train_steps=15, gan_noise_dim=1,
                          start_update_thr=False, run_id="diag_underground",
                          metrics_save_rate=20)

    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    guesser.init("../../dataset/diag_underground.txt",
                 init_models=True, init_scan_batch_num=1, scan_offset=6)

    scan_idx = 1000
    scans = guesser.getScans()[scan_idx:scan_idx + scan_seq_size]
    cmdvs = guesser.cmdVel()[scan_idx:scan_idx + scan_seq_size]
    ts = guesser.timesteps()[scan_idx:scan_idx + scan_seq_size]
    scan_guessed = guesser.getScans()[scan_idx + scan_seq_size + scan_generation_step]

    gscan, _, hp = guesser.generateScan(scans, cmdvs, ts)

    nsteps = 50
    for i in range(nsteps):
        if guesser.simStep():
            if i % int(0.33*nsteps) == 0:
                gscan, _, _ = guesser.generateScan(scans, cmdvs, ts)
                guesser.plotScan(gscan)

    gscan, dscan, hp = guesser.generateScan(scans, cmdvs, ts)
    guesser.plotScan(scan_guessed, dscan[0])
    guesser.plotScan(gscan)
    guesser.plotProjection(scan_guessed, gen_params=hp)

    import matplotlib.pyplot as plt
    plt.show()
