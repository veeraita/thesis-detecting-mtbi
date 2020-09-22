# -*- coding: utf-8 -*-

import os
import sys
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
import matplotlib.pyplot as plt
import numpy as np


def fit_ica(raw, fmin=1., fmax=40., tmax=120., n_components=0.95, method='fastica', max_iter=200, decim=None,
            reject=None):
    """Filter data and fit the ICA model"""
    raw.crop(tmax=tmax)
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=fmin, h_freq=fmax, n_jobs=2)

    ica = ICA(n_components=n_components, method=method, max_iter=max_iter, random_state=97)
    # pick only MEG channels
    picks = mne.pick_types(filt_raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
    ica.fit(filt_raw, picks=picks, decim=decim, reject=reject)
    return ica


def get_excludes(ica, raw, ecg_method='ctps'):
    """Identify bad components by analyzing latent sources"""
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw, method=ecg_method)
    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw)
    except RuntimeError as e:
        print('Error with EOG detection, msg: {}'.format(e))
        eog_inds, eog_scores = [], None
    return ecg_inds, ecg_scores, eog_inds, eog_scores


def plot_ica_results(ica, raw, ecg_inds, ecg_scores, eog_inds, eog_scores, t_offset=40., show=False):
    """Assess component selection and unmixing quality by visualization"""
    raw_crop = raw.copy()
    raw_crop = raw_crop.crop(tmin=t_offset)
    raw_crop.load_data()
    figs = []
    captions = []

    #===== Plot scores =====#
    title = 'ICA scores, %s'

    if ecg_scores is not None and len(ecg_scores) > 0:
        fig_ecg_scores = ica.plot_scores(ecg_scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg', show=show)
        figs.append(fig_ecg_scores)
        captions.append('ICA component "ECG match" scores')

    if eog_scores is not None and len(eog_scores) > 0:
        fig_eog_scores = ica.plot_scores(eog_scores, exclude=eog_inds, title=title % 'eog', labels='eog', show=show)
        figs.append(fig_eog_scores)
        captions.append('ICA component "EOG match" scores')

    # ===== Plot sources =====#
    title = 'Sources related to %s artifacts (red)'

    if ecg_scores is not None and len(ecg_scores) > 0:
        ecg_show_picks = np.abs(ecg_scores).argsort()[::-1][:5]
        for i in ecg_inds:
            if i not in ecg_show_picks:
                ecg_show_picks = np.insert(ecg_show_picks, 0, i)[:-1]
    else:
        ecg_show_picks = None
    ica.exclude = ecg_inds
    fig_ecg_sources = ica.plot_sources(raw_crop, picks=ecg_show_picks, title=title % 'ecg', show=show)
    figs.append(fig_ecg_sources)
    captions.append('ICs applied to raw data, with ECG matches highlighted')

    if eog_scores is not None and len(eog_scores) > 0:
        eog_show_picks = np.abs(eog_scores).argsort()[::-1][:5]
        for i in eog_inds:
            if i not in eog_show_picks:
                eog_show_picks = np.insert(eog_show_picks, 0, i)[:-1]
    else:
        eog_show_picks = None
    ica.exclude = eog_inds
    fig_eog_sources = ica.plot_sources(raw_crop, picks=eog_show_picks, title=title % 'eog', show=show)
    figs.append(fig_eog_sources)
    captions.append('ICs applied to raw data, with EOG matches highlighted')

    # ===== Plot overlay =====#
    fig_ecg_overlay = ica.plot_overlay(raw_crop, exclude=ecg_inds, show=show)
    figs.append(fig_ecg_overlay)
    captions.append('Overlay of the original signal against the signal with ECG ICs excluded')

    fig_eog_overlay = ica.plot_overlay(raw_crop, exclude=eog_inds, show=show)
    figs.append(fig_eog_overlay)
    captions.append('Overlay of the original signal against the signal with EOG ICs excluded')

    ecg_evoked = create_ecg_epochs(raw).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))

    try:
        eog_evoked = create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
    except RuntimeError:
        eog_evoked = None

    # ===== Plot averaged ECG/EOG epochs =====#
    ica.exclude = ecg_inds
    fig_ecg_evoked = ica.plot_sources(ecg_evoked, show=show)
    figs.append(fig_ecg_evoked)
    captions.append('ICs applied to the averaged ECG epochs, with ECG matches highlighted')

    if eog_evoked:
        ica.exclude = eog_inds
        fig_eog_evoked = ica.plot_sources(eog_evoked, show=show)
        figs.append(fig_eog_evoked)
        captions.append('ICs applied to the averaged EOG epochs, with EOG matches highlighted')
    return figs, captions


def create_report(subj, task, raw_fname, report_fname, figs=None, captions=None):
    title = f'ICA results for subject {subj} ({task})'
    report = mne.Report(verbose=True, subject=subj, info_fname=raw_fname, title=title)
    report.add_figs_to_section(figs, captions)
    report.save(report_fname, open_browser=False, overwrite=True)


def main(subj, task, raw_fname, output_dir, overwrite=False, show=False):
    ica_dir = os.path.join(output_dir, subj, 'ica')
    os.makedirs(ica_dir, exist_ok=True)

    out_fname = os.path.join(ica_dir, f'{subj}-{task}-ica.fif')
    if os.path.isfile(out_fname) and not overwrite:
        print("ICA solution exists; run with overwrite=True to make a new ICA solution")
        return

    try:
        raw = mne.io.Raw(raw_fname)
    except FileNotFoundError:
        print(f"ERROR: {raw_fname} not found!")
        with open(os.path.join(output_dir, 'not_found.txt'), 'a+') as f:
            f.write(subj + ' ' + task + '\n')
        return

    ica = fit_ica(raw, decim=3, reject=dict(mag=5e-12, grad=5000e-13))
    ecg_inds, ecg_scores, eog_inds, eog_scores = get_excludes(ica, raw)
    print("ECG ICs:", ecg_inds)
    print("EOG ICs:", eog_inds)

    if ecg_inds is not None and len(ecg_inds) == 0:
        print("No ECG matches found!")
        with open(os.path.join(output_dir, 'no_ecg_matches.txt'), 'a+') as f:
            f.write(subj + ' ' + task + '\n')
    if eog_inds is not None and len(eog_inds) == 0:
        print("No EOG matches found!")
        with open(os.path.join(output_dir, 'no_eog_matches.txt'), 'a+') as f:
            f.write(subj + ' ' + task + '\n')

    n_max_ecg, n_max_eog = 2, 1

    if ecg_inds is not None and len(ecg_inds) > n_max_ecg:
        ecg_inds = ecg_inds[:n_max_ecg]
    if eog_inds is not None and len(eog_inds) > n_max_eog:
        eog_inds = eog_inds[:n_max_eog]
    if np.array(eog_scores).ndim == 2:
        eog_scores = eog_scores[0]

    report_fname = os.path.join(ica_dir, f'{subj}-{task}-ica-report.html')
    figs, captions = plot_ica_results(ica, raw, ecg_inds, ecg_scores, eog_inds, eog_scores, show=show)
    create_report(subj, task, raw_fname, report_fname, figs, captions)

    ica.exclude = ecg_inds
    ica.exclude += eog_inds

    ica.save(out_fname)


if __name__ == "__main__":
    main(*sys.argv[1:], overwrite=True, show=True)
