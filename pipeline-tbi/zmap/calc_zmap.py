# -*- coding: utf-8 -*-

import os
import sys
import mne
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from visualize.visualize import visualize_stc


def calc_zmap(stc_fpath, avg_stc_fpath, var_stc_fpath, outfile):
    sub_stc = mne.read_source_estimate(stc_fpath, 'fsaverage')
    avg_stc = mne.read_source_estimate(avg_stc_fpath, 'fsaverage')
    var_stc = mne.read_source_estimate(var_stc_fpath, 'fsaverage')
    sd = np.sqrt(var_stc.data)

    t_data = (sub_stc.data - avg_stc.data) / sd
    zmap_stc = sub_stc.copy()
    zmap_stc.data = t_data

    zmap_stc.save(outfile)

    data_outfile = outfile + '-data.csv'
    np.savetxt(data_outfile, t_data, fmt='%.5f', delimiter=",")

    return zmap_stc


def plot_zmap(zmap_stc_fname, subject, fig_fname, subjects_dir):
    visualize_stc(zmap_stc_fname, fig_fname, subjects_dir, subject, colormap='coolwarm', clim=dict(kind='value', pos_lims=[0.0, 1.0, 2.0]))


def main(subject, task, data_dir, subjects_dir, averages_dir, cohorts='age', window=True):
    if cohorts is not None:
        raw_fname = os.path.join(data_dir, subject, 'ica', f'{subject}-{task}-ica-recon.fif')
        raw = mne.io.Raw(raw_fname)
        birthyear = raw.info['subject_info']['birthday'][0]
        meas_year = raw.info['meas_date'].year
        age = meas_year - birthyear
        cohort_idx = int((age - 8) / 10)
        print('Cohort:', cohort_idx)

        avg_stc_fpath = os.path.join(averages_dir, f'avg-cohort-{cohort_idx}-camcan')
        var_stc_fpath = os.path.join(averages_dir, f'var-cohort-{cohort_idx}-camcan')
        if cohorts == 'random':
            avg_stc_fpath += '-random'
            var_stc_fpath += '-random'
    else:
        avg_stc_fpath = os.path.join(averages_dir, f'avg-camcan')
        var_stc_fpath = os.path.join(averages_dir, f'var-camcan')

    outdir = os.path.join(data_dir, subject, 'zmap')
    os.makedirs(outdir, exist_ok=True)

    if window:
        for i in range(40, 390, 50):
            stc_fpath = os.path.join(data_dir, subject, 'psd', f'{subject}-{task}-{i}-psd-fsaverage')
            if cohorts is not None:
                zmap_outfile = os.path.join(outdir, f'{subject}-{task}-{i}-{cohorts}-psd-zmap')
                fig_fpath = os.path.join(data_dir, subject, 'fig', f'{subject}-{task}-{i}-{cohorts}-cohort-psd-zmap.png')
            else:
                zmap_outfile = os.path.join(outdir, f'{subject}-{task}-{i}-psd-zmap')
                fig_fpath = os.path.join(data_dir, subject, 'fig', f'{subject}-{task}-{i}-psd-zmap.png')
            zmap = calc_zmap(stc_fpath, avg_stc_fpath, var_stc_fpath, zmap_outfile)
            plot_zmap(zmap_outfile, subject, fig_fpath, subjects_dir)
    else:
        stc_fpath = os.path.join(data_dir, subject, 'psd', f'{subject}-{task}-full-psd-fsaverage')
        if cohorts is not None:
            zmap_outfile = os.path.join(outdir, f'{subject}-{task}-{cohorts}-psd-zmap')
            fig_fpath = os.path.join(data_dir, subject, 'fig', f'{subject}-{task}-{cohorts}-psd-zmap.png')
        else:
            zmap_outfile = os.path.join(outdir, f'{subject}-{task}-psd-zmap')
            fig_fpath = os.path.join(data_dir, subject, 'fig', f'{subject}-{task}-psd-zmap.png')
        zmap = calc_zmap(stc_fpath, avg_stc_fpath, var_stc_fpath, zmap_outfile)
        plot_zmap(zmap_outfile, subject, fig_fpath, subjects_dir)


if __name__ == '__main__':
    subject = sys.argv[1]
    task = sys.argv[2]
    data_dir = sys.argv[3]
    subjects_dir = sys.argv[4]
    averages_dir = sys.argv[5]
    if len(sys.argv) > 6:
        cohorts = sys.argv[6]
    else:
        cohorts = None

    main(subject, task, data_dir, subjects_dir, averages_dir, cohorts=cohorts, window=True)

