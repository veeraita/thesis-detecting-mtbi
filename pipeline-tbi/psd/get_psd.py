# -*- coding: utf-8 -*-

import mne
import os
import sys
import glob

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from visualize.visualize import *

spacing = 4


def create_source_space(subj, src_fname, subjects_dir):
    try:
        src = mne.read_source_spaces(src_fname)
    except:
        src = mne.setup_source_space(subj, spacing=f'ico{spacing}', subjects_dir=subjects_dir)
        mne.write_source_spaces(src_fname, src)

    return src


def make_bem_solution(subj, bemmodel_fname, bemsolution_fname, subjects_dir, force_calculate=False):
    if os.path.exists(bemmodel_fname) and not force_calculate:
        model = mne.read_bem_surfaces(bemmodel_fname)
    else:
        model = mne.make_bem_model(subj, ico=spacing, conductivity=[0.3], subjects_dir=subjects_dir)
        mne.write_bem_surfaces(bemmodel_fname, model)

    if os.path.exists(bemsolution_fname) and not force_calculate:
        bem_sol = mne.read_bem_solution(bemsolution_fname)
    else:
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bemsolution_fname, bem_sol)
    return bem_sol


def make_inverse_model(inv_fname, raw_fname, cov_fname, trans_fname, src, bem_sol, force_calculate=False):
    if os.path.exists(inv_fname) and not force_calculate:
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    else:
        raw = mne.io.Raw(raw_fname)
        cov = mne.read_cov(cov_fname)
        fwd = mne.make_forward_solution(raw.info, trans_fname, src, bem_sol)
        inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2)
        mne.minimum_norm.write_inverse_operator(inv_fname, inv)

    return inv


def make_source_psd(raw_fname, stc_fname, fsaverage_fname, fsaverage_src_fname, inv, subj, subjects_dir, force_calculate=False):
    raw = mne.io.Raw(raw_fname)
    if os.path.exists(stc_fname + '-lh.stc') and not force_calculate:
        stc = mne.read_source_estimate(stc_fname)
    else:
        stc = mne.minimum_norm.compute_source_psd(raw, inv, lambda2=0.1111111111111111, method='dSPM', tmin=None, \
                                                  tmax=None, fmin=1.0, fmax=40.0, n_fft=2048 * 4, overlap=0.5,
                                                  pick_ori=None, label=None, nave=1, pca=True, \
                                                  prepared=False)
        stc.save(stc_fname)

    src_to = mne.read_source_spaces(fsaverage_src_fname)
    fsave_vertices = [s['vertno'] for s in src_to]
    morph = mne.compute_source_morph(stc, subject_from=subj, subject_to='fsaverage', subjects_dir=subjects_dir,
                                     spacing=fsave_vertices, verbose=True)
    stc_fsaverage = morph.apply(stc)
    stc_fsaverage.save(fsaverage_fname)


def process_subject(subj, subjects_dir, output_dir, tasks=['EC'], force_calculate=False):
    scr_dir = os.path.join(output_dir, subj, 'src')
    bem_dir = os.path.join(output_dir, subj, 'bem')
    inv_dir = os.path.join(output_dir, subj, 'inv')
    psd_dir = os.path.join(output_dir, subj, 'psd')
    fig_dir = os.path.join(output_dir, subj, 'fig')

    for d in [scr_dir, bem_dir, inv_dir, psd_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    src_fname = os.path.join(scr_dir, f'{subj}-ico{spacing}-src.fif')
    fsaverage_src_fname = f'/scratch/nbe/tbi-meg/veera/processed/fsaverage/src/fsaverage-ico{spacing}-src.fif'
    bemmodel_fname = os.path.join(bem_dir, f'{subj}-bem.fif')
    bemsolution_fname = os.path.join(bem_dir, f'{subj}-bem-sol.fif')
    cov_fname = os.path.join(output_dir, subj, 'cov', f'{subj}-cov.fif')

    src = create_source_space(subj, src_fname, subjects_dir)
    visualize_source_space(src_fname, os.path.join(fig_dir, f'{subj}-src.png'), subjects_dir, subj)

    bem_sol = make_bem_solution(subj, bemmodel_fname, bemsolution_fname, subjects_dir, force_calculate)
    visualize_bem(os.path.join(fig_dir, f'{subj}-bem.png'), subjects_dir, subj, src)

    for task in tasks:
        for i in range(40, 390, 50):
            raw_fname = os.path.join(output_dir, subj, 'ica', f'{subj}-{task}-{i}-ica-recon.fif')
            #raw_fname = f'/scratch/work/italinv1/tbi/meg/{subject}_{task}_tsss_mc.fif'
            trans_fname = os.path.join(output_dir, subj, 'trans', f'{subj}-{task}-{i}-new-hs-AR-trans.fif')
            inv_fname = os.path.join(inv_dir, f'{subj}-{task}-{i}-inv.fif')
            stc_fname = os.path.join(psd_dir, f'{subj}-{task}-{i}-psd-dSPM')
            fsaverage_fname = os.path.join(psd_dir, f'{subj}-{task}-{i}-psd-fsaverage')

            inv = make_inverse_model(inv_fname, raw_fname, cov_fname, trans_fname, src, bem_sol, force_calculate)
            make_source_psd(raw_fname, stc_fname, fsaverage_fname, fsaverage_src_fname, inv, subj, subjects_dir, force_calculate)
            visualize_stc(fsaverage_fname, os.path.join(fig_dir, f'{subj}-{task}-{i}-stc-fsaverage.png'), subjects_dir, subj)
            visualize_psd(fsaverage_fname, os.path.join(fig_dir, f'{subj}-{task}-{i}-psd-fsaverage.png'), subj)


if __name__ == "__main__":
    subject = sys.argv[1]
    subjects_dir = sys.argv[2]
    output_dir = sys.argv[3]
    if 'camcan' in subjects_dir:
        tasks = ['rest']
    else:
        tasks = ['EC']
    process_subject(subject, subjects_dir, output_dir, tasks=tasks, force_calculate=True)
