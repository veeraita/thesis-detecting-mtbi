# -*- coding: utf-8 -*-

import os
import sys
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
import matplotlib.pyplot as plt
import numpy as np


def fit_ica(raw, fmin=1., fmax=None, n_components=0.95, method='fastica', max_iter=200, decim=None):
    raw.crop(tmax=60.)
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=fmin, h_freq=fmax)

    ica = ICA(n_components=n_components, method=method, max_iter=max_iter, random_state=97)
    picks = mne.pick_types(filt_raw.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
    ica.fit(filt_raw, picks=picks, decim=decim)
    return ica


def get_excludes(ica, raw, ecg_method='correlation'):
    ecg_inds, ecg_scores = ica.find_bads_ecg(raw, method=ecg_method)
    try:
        eog_inds, eog_scores = ica.find_bads_eog(raw)
    except RuntimeError as e:
        print('Error with EOG detection, msg: {}'.format(e))
        eog_inds, eog_scores = None, None
    return ecg_inds, ecg_scores, eog_inds, eog_scores


def plot_ica_results(ica, raw, ecg_inds, ecg_scores, eog_inds, eog_scores, t_offset=20., show=False):
    raw_crop = raw.copy()
    raw_crop = raw_crop.crop(tmin=t_offset)
    raw_crop.load_data()
    figs = []
    captions = []

    #===== Plot scores =====#
    title = 'ICA scores, %s'

    fig_ecg_scores = ica.plot_scores(ecg_scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg', show=show)
    figs.append(fig_ecg_scores)
    captions.append('ICA component "ECG match" scores')

    fig_eog_scores = ica.plot_scores(eog_scores, exclude=eog_inds, title=title % 'eog', labels='eog', show=show)
    figs.append(fig_eog_scores)
    captions.append('ICA component "EOG match" scores')

    # ===== Plot sources =====#
    title = 'Sources related to %s artifacts (red)'

    ecg_show_picks = np.abs(ecg_scores).argsort()[::-1][:5]
    ica.exclude = ecg_inds
    fig_ecg_sources = ica.plot_sources(raw_crop, picks=ecg_show_picks, title=title % 'ecg', show=show,
                                       start=20)
    figs.append(fig_ecg_sources)
    captions.append('ICs applied to raw data, with ECG matches highlighted')

    eog_show_picks = np.abs(eog_scores).argsort()[::-1][:5]
    ica.exclude = eog_inds
    fig_eog_sources = ica.plot_sources(raw_crop, picks=eog_show_picks, title=title % 'eog', show=show,
                                       start=20)
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

    eog_evoked = create_eog_epochs(raw).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))

    # ===== Plot averaged ECG/EOG epochs =====#
    ica.exclude = ecg_inds
    fig_ecg_evoked = ica.plot_sources(ecg_evoked, show=show)
    figs.append(fig_ecg_evoked)
    captions.append('ICs applied to the averaged ECG epochs, with ECG matches highlighted')

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


def main(subj, task, raw_fname, output_dir):
    ica_dir = os.path.join(output_dir, subj, 'ica')
    os.makedirs(ica_dir, exist_ok=True)

    raw = mne.io.Raw(raw_fname)
    ica = fit_ica(raw, decim=None)
    ecg_inds, ecg_scores, eog_inds, eog_scores = get_excludes(ica, raw)
    print("ECG ICs:", ecg_inds)
    print("EOG ICs:", eog_inds)

    if len(ecg_inds) == 0:
        print("No ECG matches found!")
        with open(os.path.join(output_dir, 'no_ecg_matches.txt'), 'a+') as f:
            f.write(subj + ' ' + task + '\n')
    if len(eog_inds) == 0:
        print("No EOG matches found!")
        with open(os.path.join(output_dir, 'no_eog_matches.txt'), 'a+') as f:
            f.write(subj + ' ' + task + '\n')

    report_fname = os.path.join(ica_dir, f'{subj}-{task}-ica-report.html')
    figs, captions = plot_ica_results(ica, raw, ecg_inds, ecg_scores, eog_inds, eog_scores)
    create_report(subj, task, raw_fname, report_fname, figs, captions)

    ica.exclude = ecg_inds
    ica.exclude += eog_inds

    out_fname = os.path.join(ica_dir, f'{subj}-{task}-ica.fif')
    ica.save(out_fname)


if __name__ == "__main__":
    main(*sys.argv[1:])
