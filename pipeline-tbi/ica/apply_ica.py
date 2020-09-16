# -*- coding: utf-8 -*-

import os
import sys
import mne
from mne.preprocessing import read_ica


def main(subj, task, raw_fname, output_dir):
    ica_dir = os.path.join(output_dir, subj, 'ica')
    os.makedirs(ica_dir, exist_ok=True)

    ica_fname = os.path.join(ica_dir, f'{subj}-{task}-ica.fif')
    ica = read_ica(ica_fname)

    raw = mne.io.Raw(raw_fname)
    raw.load_data()
    ica.apply(raw)

    out_fname = os.path.join(ica_dir, f'{subj}-{task}-ica-recon.fif')
    raw.save(out_fname)


if __name__ == "__main__":
    main(*sys.argv[1:])