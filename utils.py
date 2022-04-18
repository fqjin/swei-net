"""
Copyright 2022 Felix Q. Jin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from scipy.interpolate import interp2d
from scipy.io import loadmat, savemat
from scipy.signal import hilbert
default_sws = 2.1
default_shape = (16, 100)
default_dxdt = 0.423077 / 0.179998


def imagesc(x, y, data, **kwargs):
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    extent = [x[0]-dx/2, x[-1]+dx/2, y[-1]+dy/2, y[0]-dy/2]
    plt.imshow(data, extent=extent, **kwargs)


def load_data(file):
    """Customize this function for your specific application"""
    data = loadmat(file)
    displ = data['displ']
    x = data['xMm'].ravel()
    t = data['tMsec'].ravel()
    return t, x, displ


def preprocess_data(t, x, displ, phase_shift=False, expected_sws=None):
    """Preprocesses data for input into SweiNet
    
    Args:
        t: time vector
        x: space vector
        displ: a single (2D) or stack (3D) of spacetime planes
        phase_shift: converts velocity data to displacement-like appearance.
            Default is False.
        expected_sws: optional, if known, adjusts scaling to compensate
    """
    displ = displ.copy()
    if len(displ.shape) == 2:
        displ = displ[None]
    elif len(displ.shape) != 3:
        raise ValueError('displ shape is not 2 or 3')
    if len(x) != displ.shape[1]:
        raise ValueError('x shape does not match displ')
    if len(t) != displ.shape[2]:
        raise ValueError('t shape does not match displ')

    if phase_shift:
        displ = hilbert(displ).imag

    # Resize spatial to default x size
    xsize = default_shape[0]
    if len(x) > xsize:
        x = cv2.resize(x[None], (xsize, 1))[0]
        displ = np.stack([cv2.resize(d, (d.shape[1], xsize)) for d in displ])
    elif len(x) < xsize:
        x = cv2.resize(x[None], (xsize+2, 1))[0, 1:-1]
        displ = np.stack([cv2.resize(d, (d.shape[1], xsize+2))[1:-1] for d in displ])

    dx = np.mean(np.diff(x)).item()
    dt = np.mean(np.diff(t)).item()
    dxdt = dx / dt

    # Resize temporal to achieve target dxdt
    if expected_sws is None:
        tsize = default_shape[1]
    else:
        tsize = round(len(t) * default_dxdt/dxdt * expected_sws/default_sws)
    if len(t) > tsize:
        t = cv2.resize(t[None], (tsize, 1))[0]
        displ = np.stack([cv2.resize(d, (tsize, d.shape[0])) for d in displ])
    elif len(t) < tsize:
        t = cv2.resize(t[None], (tsize+2, 1))[0, 1:-1]
        displ = np.stack([cv2.resize(d, (tsize+2, d.shape[0]))[:, 1:-1] for d in displ])
        
    dt = np.mean(np.diff(t))
    dxdt = dx / dt

    # Pad or crop to default t size
    if tsize < default_shape[1]:
        # minimum padding becomes zeros after normalization
        displ = np.pad(displ, ((0,0), (0,0), (0,default_shape[1]-tsize)), mode='minimum')
    else:
        displ = displ[:, :, :default_shape[1]]


    # Normalize to [0, 1]
    displ -= np.min(displ, 2, keepdims=True)
    displ /= np.max(displ, 2, keepdims=True)

    return {'displ': displ,
            'dxdt_factor': dxdt / default_dxdt}


def make_sim_data_1():
    """Generates simulated data similar to cervix training data"""
    np.random.seed(1234)
    t = np.linspace(0.07, 17.89, 100)[None, :]
    x = np.linspace(2.12, 8.46, 16)[:, None]
    t_ = t - 1.0
    x_ = x / 2.0
    d = (x * np.exp(-0.5*x)) * np.cos(t_-x_) * np.exp(-0.2 * (t_-x_ + 0.2 + x/8)**2)
    d += 0.015 * np.real(ifft2(np.abs(fft2(d))**0.6 * fft2(np.random.randn(*d.shape))))
    savemat('sim_data_1.mat', {'xMm': x, 'tMsec': t, 'displ': d})


def make_sim_data_2():
    """Generates simulated data different from training data
    but similar to the external data from the SweiNet paper
    """
    np.random.seed(1234)
    t = np.linspace(0.86, 7.00, 310)[None, :]
    x = np.linspace(5.10, 9.98, 34)[:, None]
    t_ = t - 0.7
    x_ = x / 5.0
    d = 0.1 * (2.75-x_) * np.cos(2*(t_-x_)) * np.exp(-.1*(t_-x_-0.5)**4)
    d += 0.15 - 0.05*t
    noise = 0.03 * np.random.randn(len(x), len(t[0,::51]))
    d += interp2d(t[0,::51], x[:,0], noise, kind='cubic', bounds_error=False)(t[0], x[:,0])
    savemat('sim_data_2.mat', {'xMm': x, 'tMsec': t, 'displ': d})
