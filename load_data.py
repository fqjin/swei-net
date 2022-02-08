"""
Copyright 2021 Felix Q. Jin

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
from scipy.io import loadmat
def_shape = (16, 100)
def_dx = 0.423077
def_dt = 0.179998
def_dxdt = def_dx / def_dt


def load_data(file, target_dxdt_factor=None):
    """Loads SWEI data

    Customize this script for your specific application
    Returns:
        displ: N x 16 x 100 array of space-time displacement data
        dxdt_factor: scalar or N array of the ratio dxdt:default
        coords: N x M array of associated coordinates (arbitrary)
    """
    data = loadmat(file)

    data = data['arfi'][0, 0]['directionalFilt'][0, 0]
    latmm = data['swTravelDistMm'].ravel()
    tms = data['tMsec'].ravel()
    zMm = data['zMm'].ravel()
    xMm = np.mean(data['xMm'], axis=0)

    tms = tms[:100]
    latmm = np.flip(latmm)
    displ = data['data'][:, :, :100, :]  # z x 16 x 100 x 2p
    displ = np.flip(displ, axis=1)
    displ = displ.transpose((0, 3, 1, 2)).reshape((-1, 16, 100))

    dx = np.mean(np.diff(latmm))
    dt = np.mean(np.diff(tms))
    dxdt = dx / dt

    # Scale data to target dxdt
    if target_dxdt_factor is not None:
        tsize = round(len(tms) * target_dxdt_factor)
        if len(tms) > tsize:
            tms = cv2.resize(tms[None], (tsize, 1))[0]
            displ = np.stack([
                cv2.resize(d, (tsize, d.shape[0]))
                for d in displ
            ])
        elif len(tms) < tsize:
            tms = cv2.resize(tms[None], (tsize+2, 1))[0, 1:-1]
            displ = np.stack([
                cv2.resize(d, (tsize+2, d.shape[0]))[:, 1:-1]
                for d in displ
            ])

        if tsize < def_shape[1]:
            # minimum padding becomes zeros after normalization
            displ = np.pad(displ,
                           ((0, 0), (0, 0), (0, def_shape[1] - tsize)),
                           mode='minimum')
            dt = np.mean(np.diff(tms))
            dxdt = dx / dt
            # tms = np.concatenate([
            #     tms,
            #     (np.arange(def_shape[1] - tsize) + 1) * dt + tms[-1]
            # ])
        else:
            tms = tms[:def_shape[1]]
            displ = displ[:, :, :def_shape[1]]
            dt = np.mean(np.diff(tms))
            dxdt = dx / dt

    # Normalize
    displ -= np.min(displ, 2, keepdims=True)
    displ /= np.max(displ, 2, keepdims=True)

    dxdt_factor = dxdt / def_dxdt
    coords = np.stack(np.meshgrid(zMm, xMm, indexing='ij'), axis=-1)
    coords = coords.reshape((len(displ), 2))

    return displ, dxdt_factor, coords
