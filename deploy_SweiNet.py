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
import os
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.io import savemat
from tqdm import tqdm
from load_data import load_data
from network import EASINet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {}'.format(device))
torch.backends.cudnn.benchmark = True
weights = torch.load('SweiNet_weights.pt')


def construct_model(i):
    m = EASINet(out_c=2, base_c=16, c_fact=(2, 4, 4))
    m.load_state_dict(weights[i])
    m.eval()
    m.to(device)
    return m


def main(args):
    if args.saveas not in ('npz', 'csv', 'mat'):
        raise ValueError('Save format not recognized')

    files = list(Path(args.data_dir).rglob(args.filename))
    if not files:
        raise RuntimeError(f'No {args.filename} files found in data directory')
    print(f'Got {len(files)} files')

    if not args.ensemble:
        print('Not using ensemble')
        models = [construct_model(0)]
    else:
        print('Using ensemble')
        models = [construct_model(i) for i in range(1, 31)
                  if i != 18]

    for file in tqdm(files):
        displ, dxdt_factor, coords = load_data(file, args.target_dxdt_factor)
        displ = torch.from_numpy(displ).float().to(device)

        with torch.no_grad():
            z = [m(displ[:, None]) for m in models]
            z = torch.mean(torch.stack(z), dim=0)

        z[:, 0] = torch.exp(z[:, 0]) * dxdt_factor
        z[:, 1] = torch.exp(torch.exp(z[:, 1] / 2))
        z = z.cpu().numpy()

        if args.saveas == 'npz':
            np.savez(file.parent.joinpath('sweinet_out'),
                     nn_sws=z[:, 0], nn_uncert=z[:, 1],
                     coords=coords)

        elif args.saveas == 'csv':
            df = {f'coords{i}': coords[:, i] for i in range(coords.shape[1])}
            df['nn_sws'] = z[:, 0]
            df['nn_uncert'] = z[:, 1]
            df = pd.DataFrame(df)
            df.to_csv(file.parent.joinpath('sweinet_out.csv'),
                      index=False)

        elif args.saveas == 'mat':
            savemat(file.parent.joinpath('sweinet_out.mat'),
                    {'nn_sws': z[:, 0],
                     'nn_uncert': z[:, 1],
                     'coords': coords}
                    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='.',
                   help='Root directory containing data files. Default "."')
    p.add_argument('--filename', type=str, default='arfi.mat',
                   help='Name of data file. Default: arfi.mat')
    p.add_argument('--ensemble', action='store_true',
                   help='Use ensemble of models (recommended)')
    p.add_argument('--target_dxdt_factor', type=float, default=None,
                   help='Target dxdt scaling factor. Default None')
    p.add_argument('--saveas', type=str, default='npz',
                   help='Output format (npz, csv, mat)')
    main(p.parse_args())
