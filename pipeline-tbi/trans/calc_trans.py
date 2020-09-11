# -*- coding: utf-8 -*-
from mne.gui._coreg_gui import CoregModel
import os
import sys
import glob


def calcTrans(subj, input_dir, subjects_dir, output_dir):
    meg_files = glob.glob(os.path.join(input_dir, subj + '_*.fif'))
    if len(meg_files) == 0:
        print('No MEG file found, exiting')
        sys.exit()

    meg_file = meg_files[0]
    print(meg_file)

    cm = CoregModel()
    cm.mri.subject_source.set(subjects_dir=subjects_dir)
    cm.hsp.trait_set(file=meg_file)
    cm.mri.fid.trait_set(file=os.path.join(subjects_dir, subj, 'bem', subj + '-fiducials.fif'))
    cm.fit_fiducials()
    cm.omit_hsp_points(0.020)
    cm.fit_icp()

    logs_dir = os.path.join(output_dir, subj, 'coreg_logs')
    os.makedirs(logs_dir)
    with open(os.path.join(logs_dir, subj + '_hs.csv'), 'w') as f:
        #cm.print_traits()
        f.write(str(cm.lpa_distance) + '\n')
        f.write(str(cm.rpa_distance) + '\n')
        f.write(str(cm.nasion_distance) + '\n')
        f.write(str(cm.point_distance) + '\n')

    trans_dir = os.path.join(output_dir, subj, 'trans')
    os.makedirs(trans_dir)
    cm.save_trans(os.path.join(trans_dir, subj + '-new-hs-AR-trans.fif'))


if __name__ == "__main__":
    calcTrans(*sys.argv[1:])
