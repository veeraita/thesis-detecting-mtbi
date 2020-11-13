#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  21 15:30:53 2020

@author: mialil
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict


def feature_dataframe_inparcel(subjects, fname, tbi_dir, camcan_dir, parc='aparc'):
    rows = []

    for i, subject in enumerate(subjects):
        print('{}/{}, Subject: {}'.format(i + 1, len(subjects), subject))

        if subject.startswith('sub-'):
            task = 'rest'
            data_dir = camcan_dir
        else:
            task = 'EC'
            data_dir = tbi_dir

        for t in range(40, 390, 50):
            t = str(t)

            freqs = defaultdict(list)
            amps = defaultdict(list)
            widths = defaultdict(list)
            offsets = []
            exponents = []

            try:
                params = np.load(os.path.join(data_dir, fname).format(subject, subject, task, t, parc), allow_pickle=True).item()
                for label, all_params in params.items():
                    if label.startswith('unknown'): continue

                    for band in ['delta', 'theta', 'alpha']:
                        peak = all_params[f'{band}_peak_params']
                        freqs[band].append(peak[0])
                        amps[band].append(peak[1])
                        widths[band].append(peak[2])

                    # Aperiodic params: offset, (knee), exponent
                    aperiodic = all_params['aperiodic_params']

                    offsets.append(aperiodic[0])
                    exponents.append(aperiodic[-1])

                for band in ['delta', 'theta', 'alpha']:
                    rows.append([subject + '_' + t, f'{band}_freq', *freqs[band]])
                    rows.append([subject + '_' + t, f'{band}_amp', *amps[band]])
                    rows.append([subject + '_' + t, f'{band}_width', *widths[band]])
                rows.append([subject + '_' + t, 'aperiodic_offset', *offsets])
                rows.append([subject + '_' + t, 'aperiodic_exponent', *exponents])

            except OSError as e:
                print(e)

    # Create dataframe
    labels = [l for l in params.keys() if not l.startswith('unknown')]

    cols = ['subject', 'feature', *labels]
    df_features = pd.DataFrame(rows, columns=cols)
    del rows
    return df_features


def get_subject_features(df, features):
    columns_to_exclude = ('subject', 'feature')
    parcellation_labels = [c for c in df.columns if c not in columns_to_exclude]
    feature_data = [df[df.feature == bb].set_index('subject')[
                        parcellation_labels].rename(lambda s: bb + '-' + s, axis=1) for bb in features]

    meg_data = pd.concat(feature_data, axis=1, join='inner', sort=False)
    return meg_data


def get_parcel_features(df, features):
    columns_to_exclude = ('subject', 'feature')
    parcellation_labels = [c for c in df.columns if c not in columns_to_exclude]
    meg_data = df.pivot_table(columns=['feature', 'subject'], values=parcellation_labels, aggfunc='first')
    meg_data = meg_data.stack()
    meg_data.index = ['-'.join((row[1], row[0])).strip() for row in meg_data.index.values]
    return meg_data[list(features)]


def main():
    tbi_dir = '/scratch/nbe/tbi-meg/veera/processed'
    camcan_dir = '/scratch/nbe/restmeg/veera/processed'
    output_dir = '/scratch/nbe/tbi-meg/veera/fooof_data'
    subjects = sorted(
        [f.name for f in os.scandir(tbi_dir) if f.is_dir()] + [f.name for f in os.scandir(camcan_dir) if f.is_dir()])
    #subjects = ['002']

    parc = 'aparc_sub'

    # Create features based on aparc_sub parcellation
    fname = '{}/fooof/{}-{}-{}-{}-fooof-results-1Hz-knee.npy'

    df_features = feature_dataframe_inparcel(subjects, fname, tbi_dir, camcan_dir, parc=parc)
    #print(df_features.head(50))
    #print(df_features.tail(50))
    features = ('alpha_freq',
                'alpha_amp',
                'alpha_width',
                'theta_freq',
                'theta_amp',
                'theta_width',
                'delta_freq',
                'delta_amp',
                'delta_width',
                'aperiodic_offset',
                'aperiodic_exponent'
                )
    print(df_features)
    meg_data_subjects = get_subject_features(df_features, features)
    meg_data_parcels = get_parcel_features(df_features, features)
    print(meg_data_subjects)
    # Save dataframe
    feature_fname_subjects = os.path.join(output_dir, f'meg_{parc}_features_subjects_window_v2.h5')
    feature_fname_parcels = os.path.join(output_dir, f'meg_{parc}_features_parcels_window_v2.h5')
    meg_data_subjects.to_hdf(feature_fname_subjects, key=f'meg_{parc}_features_subjects_window_v2', mode='w')
    meg_data_parcels.to_hdf(feature_fname_parcels, key=f'meg_{parc}_features_parcels_window_v2', mode='w')


if __name__ == "__main__":
    main()
