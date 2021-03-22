# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = '/scratch/nbe/tbi-meg/veera/zmap-data/'
DATA_FILENAME = f'zmap_data_aparc_sub_f8_absolute.csv'
FPATH = os.path.join(DATA_DIR, DATA_FILENAME)
RESULTS_FPATH = 'reports/subject_correct_predictions.csv'
CASES = ['%03d' % n for n in range(1, 28)]



def create_data_matrix(df, names):
    data = np.zeros((len(names), 448, 40))
    for i in range(len(names)):
        data[i] = df.filter(regex=names[i], axis=0).values
    return data


def plot_heazmaps(data_mats, titles, fpath, nrows=1, ncols=1, figsize=(12, 6), colormap='RdBu_r'):
    fig = plt.figure(figsize=figsize)

    for i, (data, title) in enumerate(zip(data_mats, titles)):
        ax = fig.add_subplot(nrows, ncols, i+1)
        im = ax.imshow(data, aspect=0.1, cmap=colormap, vmin=-2, vmax=2)
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        if i % ncols == 0:
            ax.set_ylabel('Location', fontsize=12)

    fig.subplots_adjust(right=0.8, wspace=0.3)
    if ncols == 2 and nrows == 1:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    elif nrows == 2:
        cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    else:
        cbar_ax = fig.add_axes([0.85, 0.12, 0.02, 0.6])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(fpath)


def main():
    dataset = pd.read_csv(FPATH, header=None, index_col=0)
    all_subjects = list(dict.fromkeys(dataset.index.str.split(pat='_').str[0]))
    all_sample_names = list(dict.fromkeys(dataset.index.str.split(pat='-').str[0]))
    results_df = pd.read_csv(RESULTS_FPATH, header=0, names=['subject', 'score'], dtype={'subject':str})
    results_df.set_index('subject', inplace=True)
    correct = results_df[results_df['score'] > 0.5].index
    #incorrect = results_df[results_df['score'] <= 0.5].index

    case_names = [s for s in all_sample_names if s.split('_')[0] in CASES]
    control_names = [s for s in all_sample_names if s.split('_')[0] not in CASES]
    case_names_correct = [s for s in case_names if s.split('_')[0] in correct]
    case_names_incorrect = [s for s in case_names if s.split('_')[0] not in correct]
    control_names_correct = [s for s in control_names if s.split('_')[0] in correct]
    control_names_incorrect = [s for s in control_names if s.split('_')[0] not in correct]

    n_case_correct = len([s for s in all_subjects if s.split('_')[0] in CASES and s.split('_')[0] in correct])
    n_case_incorrect = len([s for s in all_subjects if s.split('_')[0] in CASES and s.split('_')[0] not in correct])
    n_control_correct = len([s for s in all_subjects if s.split('_')[0] not in CASES and s.split('_')[0] in correct])
    n_control_incorrect = len([s for s in all_subjects if s.split('_')[0] not in CASES and s.split('_')[0] not in correct])


    case_data = create_data_matrix(dataset, case_names)
    control_data = create_data_matrix(dataset, control_names)

    cases_avg = np.mean(case_data, axis=0)
    control_avg = np.mean(control_data, axis=0)

    print('Plotting patient/control averages')
    plot_heazmaps([cases_avg, control_avg], ['Patients ($n=25$)', 'Controls ($n=20$)'], 'fig/heazmap_groups.png',
                  ncols=2)

    case_data_correct = create_data_matrix(dataset, case_names_correct)
    case_data_incorrect = create_data_matrix(dataset, case_names_incorrect)
    cases_correct_avg = np.mean(case_data_correct, axis=0)
    cases_incorrect_avg = np.mean(case_data_incorrect, axis=0)
    cases_diff = cases_correct_avg - cases_incorrect_avg

    # print('Plotting cases')
    # plot_heazmaps([cases_correct_avg, cases_incorrect_avg, cases_diff],
    #               ['Correctly classified patients', 'Incorrectly classified patients', 'Difference (correct - incorrect)'],
    #               'fig/heazmap_cases.png', ncols=3, figsize=(16, 6))

    control_data_correct = create_data_matrix(dataset, control_names_correct)
    control_data_incorrect = create_data_matrix(dataset, control_names_incorrect)
    controls_correct_avg = np.mean(control_data_correct, axis=0)
    controls_incorrect_avg = np.mean(control_data_incorrect, axis=0)
    controls_diff = controls_correct_avg - controls_incorrect_avg

    # print('Plotting controls')
    # plot_heazmaps([controls_correct_avg, controls_incorrect_avg, controls_diff],
    #               ['Correctly classified controls', 'Incorrectly classified controls', 'Difference (correct - incorrect)'],
    #               'fig/heazmap_controls.png', ncols=3, figsize=(16, 6))

    print('Plotting averages by classification result')
    plot_heazmaps([cases_correct_avg, cases_incorrect_avg, cases_diff, controls_correct_avg, controls_incorrect_avg, controls_diff],
                  [f'Correctly classified patients ($n={n_case_correct}$)',
                   f'Incorrectly classified patients ($n={n_case_incorrect}$)',
                   'Difference (correct - incorrect)',
                   f'Correctly classified controls ($n={n_control_correct}$)',
                   f'Incorrectly classified controls ($n={n_control_incorrect}$)',
                   'Difference (correct - incorrect)'],
                  'fig/heazmap_results.png', nrows=2, ncols=3, figsize=(16, 10))


if __name__ == "__main__":
    main()