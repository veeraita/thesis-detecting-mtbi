# -*- coding: utf-8 -*-

import os
import sys
import mne
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from visualize.visualize import visualize_stc


def calc_tmap(subject, task, data_dir, avg_stc_fpath, var_stc_fpath, outfile):
    stc_fpath = os.path.join(data_dir, subject, 'psd', f'{subject}-{task}-psd-fsaverage')
    sub_stc = mne.read_source_estimate(stc_fpath, 'fsaverage')
    avg_stc = mne.read_source_estimate(avg_stc_fpath, 'normative-avg')
    var_stc = mne.read_source_estimate(var_stc_fpath, 'normative-var')
    sd = np.sqrt(var_stc.data)

    t_data = (sub_stc.data - avg_stc.data) / sd
    tmap_stc = sub_stc.copy()
    tmap_stc.data = t_data

    tmap_stc.save(outfile)

    data_outfile = outfile + '-data.csv'
    np.savetxt(data_outfile, t_data, fmt='%.5f', delimiter=",")

    return tmap_stc


def plot_tmap(tmap_stc_fname, subject, fig_fname, subjects_dir):
    visualize_stc(tmap_stc_fname, fig_fname, subjects_dir, subject, colormap='coolwarm', clim=dict(kind='value', pos_lims=[0.0, 1.0, 3.0]))


def main(subject, task, data_dir, subjects_dir, averages_dir):
    avg_stc_fpath = os.path.join(averages_dir, 'normative-avg-40hz')
    var_stc_fpath = os.path.join(averages_dir, 'normative-var-40hz')

    outdir = os.path.join(data_dir, subject, 'tmap')
    os.makedirs(outdir, exist_ok=True)
    tmap_outfile = os.path.join(outdir, f'{subject}-{task}-psd-tmap')

    tmap = calc_tmap(subject, task, data_dir, avg_stc_fpath, var_stc_fpath, tmap_outfile)

    fig_fpath = os.path.join(data_dir, subject, 'fig', f'{subject}-{task}-psd-tmap.png')
    plot_tmap(tmap_outfile, subject, fig_fpath, subjects_dir)


if __name__ == '__main__':
    main(*sys.argv[1:])

