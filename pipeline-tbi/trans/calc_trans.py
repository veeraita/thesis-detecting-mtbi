# -*- coding: utf-8 -*-
from mne.gui._coreg_gui import CoregModel
import os
import sys
import glob


def calc_trans(subj, meg_file, subjects_dir, output_dir, task='EO', i='full'):
    print(meg_file)

    cm = CoregModel()
    cm.mri.subject_source.trait_set(subjects_dir=subjects_dir)
    cm.hsp.trait_set(file=meg_file)
    cm.mri.fid.trait_set(file=os.path.join(subjects_dir, subj, 'bem', subj + '-fiducials.fif'))
    cm.fit_fiducials()
    cm.omit_hsp_points(0.020)
    cm.fit_icp()

    logs_dir = os.path.join(output_dir, subj, 'coreg_logs')
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, f'{subj}_{task}_hs.csv'), 'w') as f:
        #cm.print_traits()
        f.write(str(cm.lpa_distance) + '\n')
        f.write(str(cm.rpa_distance) + '\n')
        f.write(str(cm.nasion_distance) + '\n')
        f.write(str(cm.point_distance) + '\n')

    trans_dir = os.path.join(output_dir, subj, 'trans')
    os.makedirs(trans_dir, exist_ok=True)
    cm.save_trans(os.path.join(trans_dir, f'{subj}-{task}-{i}-new-hs-AR-trans.fif'))


if __name__ == "__main__":
    subject = sys.argv[1]
    subjects_dir = sys.argv[2]
    output_dir = sys.argv[3]

    window = True

    if 'camcan' in subjects_dir:
        tasks = ['rest']
    else:
        tasks = ['EC']

    for task in tasks:
        if window:
            for i in range(40, 390, 50):
               meg_file = os.path.join(output_dir, subject, 'ica', f'{subject}-{task}-{i}-ica-recon.fif')
               calc_trans(subject, meg_file, subjects_dir, output_dir, task=task, i=str(i))
        else:
            meg_file = os.path.join(output_dir, subject, 'ica', f'{subject}-{task}-ica-recon.fif')
            calc_trans(subject, meg_file, subjects_dir, output_dir, task=task)
