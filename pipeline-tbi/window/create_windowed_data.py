# -*- coding: utf-8 -*-

import os
import sys
import mne

start = 40
stop = 390
duration = 200
overlap = 50


def main(subj, task, output_dir):
    data_dir = os.path.join(output_dir, subj, 'ica')
    raw_fname = os.path.join(data_dir, f'{subj}-{task}-ica-recon.fif')

    raw = mne.io.Raw(raw_fname)
    for i in range(start, stop, overlap):
        raw_c = raw.copy()
        #try:
        raw_window = raw_c.crop(tmin=float(i), tmax=float(i+duration))
        #except ValueError:
        #    raw_window = raw_c.crop(tmin=float(i))
        raw_window_fname = os.path.join(data_dir, f'{subj}-{task}-{i}-ica-recon.fif')
        raw_window.save(raw_window_fname, overwrite=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
