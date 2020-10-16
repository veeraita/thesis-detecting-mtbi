# -*- coding: utf-8 -*-

import os
import sys
import mne
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from visualize.visualize import *


def calc_relative_psd(subj, output_dir, tasks=['EC']):
    psd_dir = os.path.join(output_dir, subj, 'psd')
    fig_dir = os.path.join(output_dir, subj, 'fig')

    for task in tasks:
        fsaverage_fname = os.path.join(psd_dir, f'{subj}-{task}-psd-fsaverage')
        stc = mne.read_source_estimate(fsaverage_fname)
        relative_psd = stc.data / np.sum(stc.data, axis=1)[:,None]
        new_stc = stc.copy()
        new_stc.data = relative_psd

        outfile = os.path.join(psd_dir, f'{subj}-{task}-relative-psd-fsaverage')
        new_stc.save(outfile)
        visualize_psd(outfile, os.path.join(fig_dir, f'{subj}-{task}-relative-psd-fsaverage.png'), subj)



def main(subject, output_dir):
    if 'restmeg' in output_dir:
        tasks = ['rest']
    else:
        tasks = ['EO', 'EC']

    calc_relative_psd(subject, output_dir, tasks=tasks)


if __name__ == '__main__':
    main(*sys.argv[1:])