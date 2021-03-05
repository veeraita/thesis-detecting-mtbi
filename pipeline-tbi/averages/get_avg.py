# -*- coding: utf-8 -*-

import os
import sys
import random
import mne
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from visualize.visualize import *

cases = ['%03d' % n for n in range(28)]
controls = ['%03d' % n for n in range(28, 48)]

camcan_dir = '/scratch/nbe/restmeg/veera/processed/'
tbi_dir = '/scratch/nbe/tbi-meg/veera/processed'
subjects_dir = '/scratch/work/italinv1/tbi/mri_recons'
averages_output_dir = '/scratch/nbe/tbi-meg/veera/averages'

random.seed(17)

def get_subjects(group='tbi'):
    all_subjects = sorted([f.name for f in os.scandir(camcan_dir) if f.is_dir()] + \
                          [f.name for f in os.scandir(tbi_dir) if f.is_dir()])

    if group == 'case':
        subjects = [s for s in all_subjects if s not in controls]
    elif group == 'control':
        subjects = [s for s in all_subjects if s in controls]
    elif group == 'tbi':
        subjects = sorted([f.name for f in os.scandir(tbi_dir) if f.is_dir()])
    elif group == 'camcan':
        subjects = sorted([f.name for f in os.scandir(camcan_dir) if f.is_dir()])
    else:
        subjects = all_subjects

    return subjects


def get_fsaverage_fname(subj):
    if subj.startswith('sub-'):
        task = 'rest'
        psd_dir = os.path.join(camcan_dir, subj, 'psd')
    else:
        task = 'EC'
        psd_dir = os.path.join(tbi_dir, subj, 'psd')

    fsaverage_fname = os.path.join(psd_dir, f'{subj}-{task}-full-psd-fsaverage')

    return fsaverage_fname


def plot_avg_psd(stc_avg, stc_var, outfile):
    x = stc_avg.times
    y = 10 * np.log10(stc_avg.data.mean(axis=0))
    dy = np.sqrt(10 * np.log10(stc_var.data.mean(axis=0)))
    plt.figure()
    plt.plot(x, y)
    plt.fill_between(x, y - dy, y + dy, color='gray', alpha=0.2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB) with SD')
    plt.title('Source Power Spectrum (PSD)')
    plt.savefig(outfile)


def get_avg_psd(subjs):
    j = 1
    fsaverage_fname = get_fsaverage_fname(subjs[0])
    while not os.path.exists(fsaverage_fname + '-lh.stc'):
        fsaverage_fname = get_fsaverage_fname(subjs[j])
        j += 1

    stc_avg = mne.read_source_estimate(fsaverage_fname, 'fsaverage')
    numS = 1
    Ex = stc_avg - stc_avg
    Ex2 = stc_avg - stc_avg
    K = stc_avg.copy()

    for s in subjs[j:]:
        print(s)
        try:
            fsaverage_fname = get_fsaverage_fname(s)
            stc_c = mne.read_source_estimate(fsaverage_fname, 'fsaverage')

            stc_avg += stc_c
            
            Ex += stc_c - K
            Ex2 += (stc_c - K)*(stc_c - K)
            
            numS += 1
            sys.stdout.flush()
        except OSError as e:
            print(e)

    print("Found source estimates for", str(numS), "subjects")
    stc_avg = stc_avg / numS
    stc_var = (Ex2 - Ex * Ex / numS) / (numS - 1)
    print('Average and variance calculated')
    return stc_avg, stc_var


def get_cohorts(subjects):
    cohorts = defaultdict(list)

    for subj in subjects:
        if subj.startswith('sub-CC'):
            cohort_idx = int(subj[6])
        else:
            try:
                raw_fname = os.path.join(tbi_dir, subj, 'ica', f'{subj}-EC-ica-recon.fif')
                raw = mne.io.Raw(raw_fname)
            except FileNotFoundError:
                continue
            try:
                birthyear = raw.info['subject_info']['birthday'][0]
            except KeyError:
                print("No birthday data found")
                continue
            meas_year = raw.info['meas_date'].year
            age = meas_year - birthyear

            cohort_idx = int((age - 8) / 10)
        print("Subject:", subj, "Cohort:", cohort_idx)
        cohorts[cohort_idx].append(subj)
    return cohorts


def get_cohort(cohorts, idx, group='tbi', random_cohorts=False):
    """Calculate averages by age group"""
    if idx < 1 or idx > 7:
        print('Please use index between 1 and 7')
        return
    cohort_subjects = cohorts.get(idx, None)
    if not cohort_subjects:
        print('No subjects in cohort', idx)
        return

    group_avg_outdir = os.path.join(averages_output_dir, group)
    fig_dir = os.path.join(group_avg_outdir, 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    if random_cohorts:
        n = len(cohort_subjects)
        all_subjects = [s for sublist in cohorts.values() for s in sublist]
        subjects = random.sample(all_subjects, n)
        psd_fig_fname = os.path.join(fig_dir, f'cohort-{idx}-{group}-avg-random-psd.png')
        stc_fig_fname = os.path.join(fig_dir, f'cohort-{idx}-{group}-avg-random-stc.png')
        stc_avg_fname = os.path.join(group_avg_outdir, f'avg-cohort-{idx}-{group}-random')
        stc_var_fname = os.path.join(group_avg_outdir, f'var-cohort-{idx}-{group}-random')
    else:
        subjects = cohort_subjects
        psd_fig_fname = os.path.join(fig_dir, f'cohort-{idx}-{group}-avg-psd.png')
        stc_fig_fname = os.path.join(fig_dir, f'cohort-{idx}-{group}-avg-stc.png')
        stc_avg_fname = os.path.join(group_avg_outdir, f'avg-cohort-{idx}-{group}')
        stc_var_fname = os.path.join(group_avg_outdir, f'var-cohort-{idx}-{group}')

    stc_avg, stc_var = get_avg_psd(subjects)

    plot_avg_psd(stc_avg, stc_var, psd_fig_fname)

    stc_avg.save(stc_avg_fname)
    stc_var.save(stc_var_fname)

    visualize_stc(stc_avg_fname, stc_fig_fname, subjects_dir, 'fsaverage')


def get_all(subjects, group='tbi'):
    """Calculate average over whole dataset"""
    stc_avg, stc_var = get_avg_psd(subjects)

    group_avg_outdir = os.path.join(averages_output_dir, group)
    fig_dir = os.path.join(group_avg_outdir, 'fig')
    os.makedirs(fig_dir, exist_ok=True)

    plot_avg_psd(stc_avg, stc_var, os.path.join(fig_dir, f'{group}-avg-psd.png'))

    stc_avg_fname = os.path.join(group_avg_outdir, f'avg-{group}')
    stc_var_fname = os.path.join(group_avg_outdir, f'var-{group}')
    stc_avg.save(stc_avg_fname)
    stc_var.save(stc_var_fname)

    visualize_stc(stc_avg_fname, os.path.join(fig_dir, f'{group}-avg-stc.png'), subjects_dir, 'fsaverage')


def main(group='tbi', random_cohorts=False):
    if not group or group not in ['tbi', 'camcan', 'case', 'control']:
        group = 'tbi'
    print('Group:', group)

    subjects = get_subjects(group)
    cohorts = get_cohorts(subjects)
    print(cohorts)
    for i in range(1, 8):
        print('Starting cohort: ' + str(i))
        get_cohort(cohorts, i, group, random_cohorts=random_cohorts)

    get_all(subjects, group)

    
if __name__ == '__main__':
    main(*sys.argv[1:])

