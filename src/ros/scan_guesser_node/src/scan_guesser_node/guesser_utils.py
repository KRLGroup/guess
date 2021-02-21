
import socket

import matplotlib.pyplot as plt
import numpy as np


def println(name):
    print name


def computeTransform(cmdv):
    x, y, th = 0.0, 0.0, 0.0
    v, om, tstep = cmdv[:, 0], cmdv[:, 1], 0.033

    for n in range(cmdv.shape[0]):
        rk_th = th + 0.5*om[n]*tstep  # runge-kutta integration
        x = x + v[n]*np.cos(rk_th)*tstep
        y = y + v[n]*np.sin(rk_th)*tstep
        th = th + om[n]*tstep
    return x, y, th


def plotProjection(scan, scan_sz, scan_res, params0, params1=None, save_path=""):
    assert scan.shape[0] == scan_sz, "Wrong scan size"
    theta = scan_res*np.arange(-0.5*scan_sz, 0.5*scan_sz)
    pts = np.ones((3, scan_sz))
    pts[0] = scan*np.cos(theta)
    pts[1] = scan*np.sin(theta)

    plt.figure()
    plt.axis('equal')
    plt.plot(pts[1], pts[0], label='ref', color='#ff7f0e')

    if not params0 is None:
        x, y, th = params0[0], params0[1], params0[2]
        cth, sth = np.cos(th), np.sin(th)
        hm = np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1)))
        pts0 = np.matmul(hm, pts)
        plt.plot(pts0[1], pts0[0], label='proj', color='#1f77b4')

    if not params1 is None:
        x, y, th = params1[0], params1[1], params1[2]
        cth, sth = np.cos(th), np.sin(th)
        hm = np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1)))
        pts1 = np.matmul(hm, pts)
        plt.plot(pts1[1], pts1[0], label='pred')
    plt.legend()
    if save_path != "": plt.savefig(save_path, format='pdf')


def getScanSegments(scan, threshold):
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


def plotScan(ax, theta, scan, scan_range, sub_title):
    ax.set_theta_offset(0.5*np.pi)
    # ax.set_rmax(scan_range*1.1)
    ax.set_rlabel_position(-170)  # get radial labels away from plotted line
    segments = getScanSegments(scan, 0.95*scan_range)
    ax.plot(theta, scan, color='gray', lw=0.8, label='rick')
    for seg in segments:
        col = '#ff7f0e' if seg[2] else '#1f77b4'
        ax.plot(theta[seg[0]:seg[1]], scan[seg[0]:seg[1]],
                color=col, marker='.', markersize=0.5, lw=2)
    ax.set_title(sub_title, va='bottom')


def plotScanSequence(scans, dscans, scan_sz, scan_res, scan_range, name, save_path=""):
    assert scans.shape[1] == scan_sz, "Wrong scans size"
    assert dscans.shape[1] == scan_sz, "Wrong scans size"
    theta = scan_res*np.arange(-0.5*scan_sz, 0.5*scan_sz)

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=10)
    for s in range(scans.shape[0]):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                                 subplot_kw=dict(polar=True))
        plotScan(axes[0], theta, scans[s], scan_range, sub_title='Laser scan')
        plotScan(axes[1], theta, dscans[s], scan_range, sub_title=name)
        if save_path != "": plt.savefig(save_path + '.pdf', format='pdf')


__all__ = ['Provider', 'Receiver']


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
