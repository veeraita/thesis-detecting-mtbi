# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np


data_dir = '/scratch/nbe/tbi-meg/veera/processed'
outdir = '/scratch/nbe/tbi-meg/veera/tmap-data'
decim_freqs = 8
parc = 'aparc_sub'
labels_dir = f'/scratch/nbe/tbi-meg/veera/labels_{parc}'


def main(cohorts=None, window=True):
    subjects = sorted([f.name for f in os.scandir(data_dir) if f.is_dir()])
    labels = sorted([l.name.replace('.label', '') for l in os.scandir(labels_dir) if l.is_file()
                     and not l.name.startswith('unknown')])

    if parc is None:
        outfile = os.path.join(outdir, f'tmap_data_f{decim_freqs}.csv')
    else:
        outfile = os.path.join(outdir, f'tmap_data_{parc}_f{decim_freqs}.csv')
    if cohorts is not None:
        outfile = outfile.replace('.csv', f'_{cohorts}.csv')
    if 'restmeg' in data_dir:
        task = 'rest'
        outfile = outfile.replace('.csv', '_normative.csv')
    else:
        task = 'EC'

    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(outfile):
        print("Removing existing file")
        os.remove(outfile)

    for subject in subjects:
        tmap_dir = os.path.join(data_dir, subject, 'tmap')
        aparc_dir = os.path.join(data_dir, subject, 'parc')
        if window:
            for i in range(40, 390, 50):
                if parc is None:
                    tmap_file = os.path.join(tmap_dir, f'{subject}-{task}-{i}-psd-tmap-data.csv')
                    index = None
                else:
                    tmap_file = os.path.join(aparc_dir, f'{subject}-{task}-{i}-psd-tmap-mean-aparc-data.csv')
                    index = [f'{subject}_{i}-{label}' for label in labels]
                if cohorts is not None:
                    tmap_file = tmap_file.replace(str(i), str(i) + f'-{cohorts}')
                print(tmap_file)
                try:
                    df = pd.read_csv(tmap_file, header=None)
                except FileNotFoundError:
                    print('not found')
                    continue
                df = df[df.columns[::decim_freqs]]
                df = df.iloc[:448,:]
                if index is not None:
                    df['index'] = index
                    df = df.set_index('index')
                print(df)
                df.to_csv(outfile, mode='a', header=False)
        else:
            if parc is None:
                tmap_file = os.path.join(tmap_dir, f'{subject}-{task}-psd-tmap-data.csv')
                index = None
            else:
                tmap_file = os.path.join(aparc_dir, f'{subject}-{task}-psd-tmap-mean-aparc-data.csv')
                index = [f'{subject}-{label}' for label in labels]
            if cohorts:
                tmap_file = tmap_file.replace(task, task + f'-{cohorts}')
            print(tmap_file)
            try:
                df = pd.read_csv(tmap_file, header=None)
            except FileNotFoundError:
                continue
            df = df[df.columns[::decim_freqs]]
            df = df.iloc[:448,:]
            if index is not None:
                df['index'] = index
                df = df.set_index('index')
            print(df)
            df.to_csv(outfile, mode='a', header=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cohorts = sys.argv[1]
    else:
        cohorts = None
    main(cohorts=cohorts, window=True)

