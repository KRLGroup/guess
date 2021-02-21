#! /usr/bin/env python
# coding: utf-8

import datetime
import socket
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from scan_guesser import ScanGuesser

class Provider:
    def __init__(self, data_length, dip="127.0.0.1", dport=9559):
        self.data_length = data_length
        self.dip = dip
        self.dport = int(dport)
        self.socket = None

    def send(self, data):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.connect((self.dip, self.dport))

        assert len(data.shape) == 1, \
            'Incorrect data shape. Should be Nx1 array'
        assert data.shape[0] == self.data_length , \
            'Invalid data length. Should be (self.data_length, 1)'

        srz_data = np.array(data).tostring()
        self.socket.sendall(srz_data)
        self.socket.close()

class Receiver:
    def __init__(self, data_length, dip="127.0.0.1", dport=9559):
        self.data_length = data_length
        self.dip = dip
        self.dport = int(dport)
        self.socket = None

    def getData(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.dip, self.dport))
        self.socket.listen(1)
        c, _ = self.socket.accept()

        data = b''
        while True:
            block = c.recv(self.data_length*2)
            if not block: break
            data += block
        c.close()
        self.socket.close()

        dsrz_data = np.frombuffer(data, dtype=np.int16)
        return np.array(dsrz_data, dtype=np.float32)

if __name__ == "__main__":
    print("| ----------------------------- |")
    print("| -- ScanGuesser test-socket -- |")
    print("| ----------------------------- |\n")

    test_id='afmk'
    skt_pkg_scaling = 1000
    minibuffer_batches_num = 5
    scan_dim = 512
    clip_scans_at = 5.0
    correlated_steps = 8
    generation_step = 10
    module_rate = 1.0/30 # [1/freq]
    max_vel = 0.55  # [m/s]
    projector_max_dist = max_vel*generation_step*module_rate # [m]

    projector_lr = 1e-2

    ae_lr = 1e-2
    ae_batch_sz = 128
    ae_latent_dim = 32

    gan_discriminator_lr = 1e-3
    gan_generator_lr = 1e-3
    gan_batch_sz = 32
    gan_noise_dim = 8
    gan_smoothing_label_factor = 0.1

    metrics_save_interleave = 100
    cwd = os.path.dirname(os.path.abspath(__file__))
    dtn = datetime.datetime.now()
    dt = str(dtn.month) + "-" + str(dtn.day) + "_" + str(dtn.hour) + "-" + str(dtn.minute)
    save_path_dir = '/home/sapienzbot/ws/guess/dataset/metrics/'

    save_path_dir = os.path.join(save_path_dir, test_id + "_" + dt)
    dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), "diag_underground.txt")

    guesser = ScanGuesser(net_model="ff",  # conv; lstm
                          scan_dim=scan_dim, scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                          correlated_steps=correlated_steps, generation_step=generation_step,
                          projector_max_dist=projector_max_dist,
                          minibuffer_batches_num=minibuffer_batches_num,
                          buffer_max_size=20,
                          clip_scans_at=clip_scans_at,  # max beam length [m]
                          fit_ae=True, fit_projector=True, fit_gan=True,
                          # projector configs
                          projector_lr=projector_lr,
                          # autoencoder configs
                          ae_lr=ae_lr, ae_batch_sz=ae_batch_sz, ae_epochs=20, ae_variational=True,
                          ae_convolutional=False, ae_latent_dim=ae_latent_dim,
                          # gan configs
                          gan_batch_sz=gan_batch_sz, gan_train_steps=20, gan_noise_dim=gan_noise_dim,
                          gan_smoothing_label_factor=gan_smoothing_label_factor,
                          # run
                          start_update_thr=True, verbose=False,
                          metrics_save_path_dir=save_path_dir,
                          metrics_save_interleave=metrics_save_interleave)
    guesser.init(init_models=True)

    # 2D velocity + timestamp in seconds
    receiver = Receiver((3 + scan_dim)*correlated_steps, dport=9559)
    # generated and decoded scans + generated transform parameters
    provider = Provider(scan_dim*2 + 3, dport=9558)

    handshake_port = 9550
    print("-- Requesting modules handshake on localhost:" + str(handshake_port))
    sp = Provider(1, dport=handshake_port)
    sp.send(np.array([1]))

    print("\n-- Starting main loop...\n")
    i = 0
    while i < 10:
        i = i + 0
        try:
            data_batch_srz = receiver.getData()*(1.0/skt_pkg_scaling)
        except Exception as e:
            print("Error", str(e))
            continue

        scan_batch = data_batch_srz[:correlated_steps*scan_dim]
        scan_batch = scan_batch.reshape(correlated_steps, scan_dim)
        cmdv_batch = data_batch_srz[correlated_steps*scan_dim:]
        cmdv_batch = cmdv_batch.reshape(correlated_steps, 7)
        ts_batch = cmdv_batch[:, -1:]
        cmdv_batch = cmdv_batch[:, ::5]

        guesser.add_raw_scans(scan_batch, cmdv_batch, ts_batch)
        gscan, vscan, hp = guesser.generate_raw_scan(scan_batch, cmdv_batch, ts_batch)

        try:
            to_send = np.concatenate((gscan, vscan[-1]))
            to_send = np.concatenate((to_send, hp))
            provider.send((to_send*skt_pkg_scaling).astype(np.int16))
        except Exception as e: print("Message not sent. ", str(e))
