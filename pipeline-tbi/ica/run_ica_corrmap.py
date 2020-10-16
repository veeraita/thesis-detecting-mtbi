# -*- coding: utf-8 -*-

import os
import sys
import argparse
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import read_ica, corrmap
from run_ica import get_excludes, plot_ica_results, create_report


def get_templates(subjects, task, input_dir, output_dir):
    raws = []
    icas = []
    raw_fnames = []
    ica_fnames = []
    for subject in subjects:
        ica_dir = os.path.join(output_dir, subject, 'ica')
        if 'camcan' in input_dir:
            raw_fname = os.path.join(input_dir, subject, 'meg', f'{task}_raw_tsss_mc.fif')
        else:
            raw_fname = os.path.join(input_dir, f'{subject}_{task}_tsss_mc.fif')
        ica_fname = os.path.join(ica_dir, f'{subject}-{task}-ica.fif')
        try:
            print(f"Reading files for subject {subject}, task {task}")
            raw = mne.io.Raw(raw_fname, verbose='ERROR')
            ica = read_ica(ica_fname, verbose='ERROR')
            raws.append(raw)
            icas.append(ica)
        except FileNotFoundError:
            print("Not found")
            raws.append(None)
            icas.append(None)
        raw_fnames.append(raw_fname)
        ica_fnames.append(ica_fname)

    return raws, icas, raw_fnames, ica_fnames


def main():
    input_dir = os.environ['INPUT_DIR']
    output_dir = os.environ['OUTPUT_DIR']

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', help='The task(s) to process', nargs='*', default=None)
    parser.add_argument('template_index', help='The index to use as a template', type=int, default=0)
    parser.add_argument('artifact_type', help='Whether to use corrmap to find ecg or eog artifacts', choices=['ecg', 'eog'])
    args = parser.parse_args()

    if not args.task:
        tasks = ['EO', 'EC']
    else:
        tasks = args.task

    all_subjects = sorted([f.name for f in os.scandir(output_dir) if f.is_dir()])

    ar_type = args.artifact_type

    for task in tasks:
        raws, icas, raw_fnames, ica_fnames = get_templates(all_subjects, task, input_dir, output_dir)

        subj = all_subjects[args.template_index]
        print("Subject", subj, "chosen as template")

        template_raw = raws[args.template_index]
        template_ica = icas[args.template_index]
        if not template_raw or not template_ica:
            print("ERROR: Corrmap template not found!")
            sys.exit(1)

        print(template_ica.labels_)
        template_ica.exclude = template_ica.labels_.get(ar_type, [])
        template_ica.plot_sources(template_raw, title=f"Mark the ICs corresponding to {ar_type} artifacts")
        plt.show()
        ecg_inds = template_ica.exclude
        template_ica.labels_[f'{ar_type}/corrmap'] = ecg_inds
        existing_icas = [ica for ica in icas if ica]
        corrmap(existing_icas, template=(args.template_index, ecg_inds[0]), ch_type='mag',
                threshold=0.7, label=f'{ar_type}/corrmap', plot=False)

        for index, (raw, ica, raw_fname, ica_fname) in enumerate(zip(raws, icas, raw_fnames, ica_fnames)):
            subj = all_subjects[index]
            print(subj)
            if not raw or not ica:
                print("Data does not exist for subject", subj)
                continue

            if ar_type == 'ecg':
                corrmap_label = 'ecg/corrmap'
                manual_label = 'ecg/manual'
            elif ar_type == 'eog':
                corrmap_label = 'eog/corrmap'
                manual_label = 'eog/manual'

            if len(ica.labels_.get(corrmap_label, [])) == 0 \
                or len(ica.labels_.get(manual_label, [])) > 0 \
                or len(ica.labels_.get(ar_type, [])) > 0:
                # don't use corrmap on subjects that didn't get a new label or have been manually inspected
                print(ica.labels_)
                print("Nothing to be done for subject", subj)
                continue

            if ar_type == 'ecg':
                ecg_inds = ica.labels_.get('ecg/corrmap', [])
                eog_inds = ica.labels_.get('eog', [])
            elif ar_type == 'eog':
                ecg_inds = ica.labels_.get('ecg', [])
                eog_inds = ica.labels_.get('eog/corrmap', [])

            n_max_ecg, n_max_eog = 2, 2

            if ecg_inds is not None and len(ecg_inds) > n_max_ecg:
                ecg_inds = ecg_inds[:n_max_ecg]
            if eog_inds is not None and len(eog_inds) > n_max_eog:
                eog_inds = eog_inds[:n_max_eog]

            ica.labels_['ecg'] = ecg_inds
            ica.labels_['eog'] = eog_inds
            print(ica.labels_)

            ica.exclude = list(set(ecg_inds + eog_inds))
            print("Excluded components:", ica.exclude)

            ecg_channel_picks = mne.pick_types(raw.info, meg=False, ecg=True)
            if len(ecg_channel_picks) > 0:
                ecg_channel = raw.info['ch_names'][ecg_channel_picks[0]]
                ecg_scores = ica.score_sources(raw, ecg_channel)
            else:
                ecg_scores = None

            eog_channel_picks = mne.pick_types(raw.info, meg=False, eog=True)
            if len(eog_channel_picks) > 0:
                eog_channel = raw.info['ch_names'][eog_channel_picks[0]]
                eog_scores = ica.score_sources(raw, eog_channel)
            else:
                eog_scores = None

            ica_dir = os.path.join(output_dir, subj, 'ica')
            report_fname = os.path.join(ica_dir, f'{subj}-{task}-ica-report.html')
            figs, captions = plot_ica_results(ica, raw, ecg_inds, ecg_scores, eog_inds, eog_scores)
            create_report(subj, task, raw_fname, report_fname, figs, captions)
            ica.save(ica_fname)

            checked_file = os.path.join(output_dir, f'checked_{ar_type}_corrmap.txt')
            with open(checked_file, 'a+') as f:
                f.write(subj + ' ' + task + '\n')


if __name__ == "__main__":
    main()
