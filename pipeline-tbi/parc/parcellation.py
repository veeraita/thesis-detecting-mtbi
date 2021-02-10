import os
import sys
import mne
import numpy as np


def get_labels(subjects_dir, labels_dir, parc='aparc_sub'):
    labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=subjects_dir)
    label_files = sorted([f.path for f in os.scandir(labels_dir)])
    if len(label_files) < len(labels):
        for i, label in enumerate(labels):
            labels[i] = label.morph(subject_to='fsaverage', grade=4, subjects_dir=subjects_dir)
            labels[i].save(os.path.join(labels_dir, labels[i].name))
    else:
        labels = []
        for file in label_files:
            labels.append(mne.read_label(file, 'fsaverage'))
    return labels


def parcellate_stc(stc, labels, agg='mean'):
    parc_data = np.zeros((len(labels), stc.shape[-1]))

    for i, label in enumerate(labels):
        if label.name.startswith('unknown'):
            continue
        stc_in_label = stc.in_label(label)
        if agg == 'mean':
            parc_data[i] = np.mean(stc_in_label.data, axis=0)
        elif agg == 'max':
            parc_data[i] = np.max(stc_in_label.data, axis=0)
        else:
            raise RuntimeError('"agg" argument must be one of ("mean", "max")')

    return parc_data


def main(subj, subjects_dir, data_dir, type='absolute', ext='fsaverage', tasks=['EC'], agg='mean', cohorts=False,
         window=False):
    psd_dir = os.path.join(data_dir, subj, 'psd')
    tmap_dir = os.path.join(data_dir, subj, 'tmap')
    if ext == 'tmap':
        dir = tmap_dir
    else:
        dir = psd_dir
    for task in tasks:
        if window:
            for i in range(40, 390, 50):
                if cohorts:
                    stc_fname = os.path.join(dir, f'{subj}-{task}-{i}-{type}-cohort-psd-{ext}')
                else:
                    stc_fname = os.path.join(dir, f'{subj}-{task}-{i}-{type}-psd-{ext}')
                stc = mne.read_source_estimate(stc_fname, 'fsaverage')

                labels_dir = '/scratch/nbe/tbi-meg/veera/labels_aparc_sub'
                os.makedirs(labels_dir, exist_ok=True)
                labels = get_labels(subjects_dir, labels_dir)

                parc_stc_data = parcellate_stc(stc, labels, agg)

                outdir = os.path.join(os.path.dirname(data_dir), 'aparc_data')
                os.makedirs(outdir, exist_ok=True)
                outfile = os.path.join(outdir, os.path.basename(stc_fname) + f'-{agg}-aparc-data.csv')
                print('Saving data to', outfile)
                np.savetxt(outfile, parc_stc_data, fmt='%.7f', delimiter=",")
        else:
            if cohorts:
                stc_fname = os.path.join(dir, f'{subj}-{task}-{type}-cohort-psd-{ext}')
            else:
                stc_fname = os.path.join(dir, f'{subj}-{task}-{type}-psd-{ext}')
            stc = mne.read_source_estimate(stc_fname, 'fsaverage')

            labels_dir = '/scratch/nbe/tbi-meg/veera/labels_aparc_sub'
            os.makedirs(labels_dir, exist_ok=True)
            labels = get_labels(subjects_dir, labels_dir)

            parc_stc_data = parcellate_stc(stc, labels, agg)

            outdir = os.path.join(os.path.dirname(data_dir), 'aparc_data')
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, os.path.basename(stc_fname) + f'-{agg}-aparc-data.csv')
            print('Saving data to', outfile)
            np.savetxt(outfile, parc_stc_data, fmt='%.7f', delimiter=",")


if __name__ == "__main__":
    subject = sys.argv[1]
    subjects_dir = sys.argv[2]
    data_dir = sys.argv[3]

    if 'camcan' in subjects_dir:
        tasks = ['rest']
    else:
        tasks = ['EC']
    main(subject, subjects_dir, data_dir, type='random', ext='tmap', tasks=tasks, agg='mean', cohorts=True, window=True)
