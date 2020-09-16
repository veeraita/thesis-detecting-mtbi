# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:12:31 2018

@author: rantala2
"""

import mne
import os
import sys
import glob

from visualize import *


def create_source_space(subj, ico4_fname, subjects_dir):
    try:
        src = mne.read_source_spaces(ico4_fname)
    except:
        src = mne.setup_source_space(subj, spacing='ico4', subjects_dir=subjects_dir)
        mne.write_source_spaces(ico4_fname, src)

    return src


def make_bem_solution(subj, bemmodel_fname, bemsolution_fname, subjects_dir):
    try:
        model = mne.read_bem_surfaces(bemmodel_fname)
    except:
        model = mne.make_bem_model(subj, conductivity=[0.3], subjects_dir=subjects_dir)
        mne.write_bem_surfaces(bemmodel_fname, model)

    try:
        bem_sol = mne.read_bem_solution(bemsolution_fname)
    except:
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bemsolution_fname, bem_sol)
    return bem_sol


def make_inverse_model(inv_fname, raw_fname, cov_fname, trans_fname, src, bem_sol):
    try:
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    except:
        raw = mne.io.Raw(raw_fname)
        cov = mne.read_cov(cov_fname)
        fwd = mne.make_forward_solution(raw.info, trans_fname, src, bem_sol)
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)
        mne.minimum_norm.write_inverse_operator(inv_fname, inv)

    return inv


def make_source_psd(raw_fname, stc_fname, fsaverage_fname, inv, subj, subjects_dir):
    raw = mne.io.Raw(raw_fname)
    try:
        stc = mne.read_source_estimate(stc_fname)
    except:
        stc = mne.minimum_norm.compute_source_psd(raw, inv, lambda2=0.1111111111111111, method='dSPM', tmin=None, \
                                                  tmax=None, fmin=0.0, fmax=40.0, n_fft=2048 * 4, overlap=0.5,
                                                  pick_ori=None, label=None, nave=1, pca=True, \
                                                  prepared=False)
        stc.save(stc_fname)
    morph = mne.compute_source_morph(stc, subject_from=subj, subject_to='fsaverage', subjects_dir=subjects_dir)
    stc_fsaverage = morph.apply(stc)
    stc_fsaverage.save(fsaverage_fname)


def process_subject(subj, input_dir, subjects_dir, output_dir):
    scr_dir = os.path.join(output_dir, subj, 'src')
    bem_dir = os.path.join(output_dir, subj, 'bem')
    inv_dir = os.path.join(output_dir, subj, 'inv')
    psd_dir = os.path.join(output_dir, subj, 'psd')
    fig_dir = os.path.join(output_dir, subj, 'fig')

    for d in [scr_dir, bem_dir, inv_dir, psd_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    meg_files = glob.glob(os.path.join(input_dir, subj + '_*.fif'))
    if len(meg_files) == 0:
        print('No MEG file found, exiting')
        sys.exit()

    raw_fname = meg_files[0]
    ico4_fname = os.path.join(scr_dir, subj + '-ico4-src.fif')
    bemmodel_fname = os.path.join(bem_dir, subj + '-bem.fif')
    bemsolution_fname = os.path.join(bem_dir, subj + '-bem-sol.fif')
    trans_fname = os.path.join(output_dir, subj, 'trans', subj + '-new-hs-AR-trans.fif')
    cov_fname = os.path.join(output_dir, subj, 'cov', subj + '-cov.fif')
    inv_fname = os.path.join(inv_dir, subj + '-inv.fif')
    stc_fname = os.path.join(psd_dir, subj + '-psd-dSPM')
    fsaverage_fname = os.path.join(psd_dir, subj + '-psd-fsaverage')

    src = create_source_space(subj, ico4_fname, subjects_dir)
    visualize_source_space(ico4_fname, os.path.join(fig_dir, subj + '-src.png'), subjects_dir, subj)

    bem_sol = make_bem_solution(subj, bemmodel_fname, bemsolution_fname, subjects_dir)
    visualize_bem(os.path.join(fig_dir, subj + '-bem.png'), subjects_dir, subj)

    inv = make_inverse_model(inv_fname, raw_fname, cov_fname, trans_fname, src, bem_sol)
    make_source_psd(raw_fname, stc_fname, fsaverage_fname, inv, subj, subjects_dir)
    visualize_stc(stc_fname, os.path.join(fig_dir, subj + '-stc.png'), subjects_dir, subj)
    visualize_psd(stc_fname, os.path.join(fig_dir, subj + '-psd.png'), subj)


if __name__ == "__main__":
    process_subject(*sys.argv[1:])
