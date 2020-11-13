#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:44:27 2020

Find peaks and slopes from PSDs using fooof

@author: mhusberg
"""

import mne
import numpy as np
import os
import sys
from fooof_parametrization import fooof_inparcel

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from parc.parcellation import get_labels


def main():
    # Folders
    tbi_dir = '/scratch/nbe/tbi-meg/veera/processed'
    camcan_dir = '/scratch/nbe/restmeg/veera/processed'

    # Subjects
    if len(sys.argv) > 1:
        subjects = [sys.argv[1]]
    else:
        subjects = sorted(
            [f.name for f in os.scandir(tbi_dir) if f.is_dir()] + [f.name for f in os.scandir(camcan_dir) if
                                                                   f.is_dir()])

    # Do fooof

    # PSD
    psd_fname = '{}/psd/{}-{}-{}-psd-fsaverage'
    subjects_dir = '/m/nbe/scratch/restmeg/data/camcan/subjects_s3/'

    # Output filename
    results_fname = '{}/fooof/{}-{}-{}-{}-fooof-results-{}Hz-{}'

    # Frequency band for fitting 1/f
    fmin, fmax = 1, 40

    # Background mode for 1/f slope: 'knee' or 'fixed'
    background_mode = 'knee'

    parc = 'aparc_sub'
    labels_dir = f'/scratch/nbe/tbi-meg/veera/labels_{parc}'

    for subject in subjects:
        sys.stdout.flush()
        if subject.startswith('sub-'):
            output_dir = camcan_dir
            task = 'rest'
        else:
            output_dir = tbi_dir
            task = 'EC'
        for i in range(40, 390, 50):
            print(i)
            try:
                psd = mne.read_source_estimate(os.path.join(output_dir,
                                                            psd_fname.format(subject, subject, task, i)), subject='fsaverage')
                output_fname = os.path.join(output_dir,
                                            results_fname.format(subject, subject, task, i, parc, fmin, background_mode))
                os.makedirs(os.path.dirname(output_fname), exist_ok=True)
                labels = get_labels(subjects_dir, labels_dir, parc=parc)
                fooof_results = fooof_inparcel(subject,
                                               psd,
                                               fmin, fmax,
                                               background_mode,
                                               labels,
                                               output_fname,
                                               force_calculate=True)
            except OSError as e:
                print(e)


if __name__ == "__main__":
    main()
