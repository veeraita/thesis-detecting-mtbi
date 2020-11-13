# -*- coding: utf-8 -*-

import os
from visualize import visualize_psd

tbi_dir = '/scratch/nbe/tbi-meg/veera/processed'
camcan_dir = '/scratch/nbe/restmeg/veera/processed'

all_subjects = sorted([f.name for f in os.scandir(tbi_dir) if f.is_dir()] + [f.name for f in os.scandir(camcan_dir) if f.is_dir()])

stc_avg_fname = '/scratch/nbe/tbi-meg/veera/averages/normative/normative-avg-40hz'
stc_var_fname = '/scratch/nbe/tbi-meg/veera/averages/normative/normative-var-40hz'

for subj in all_subjects:
    if subj.startswith('sub-'):
        task = 'rest'
        psd_dir = os.path.join(camcan_dir, subj, 'psd')
        fig_dir = os.path.join(camcan_dir, subj, 'fig')
    else:
        task = 'EC'
        psd_dir = os.path.join(tbi_dir, subj, 'psd')
        fig_dir = os.path.join(tbi_dir, subj, 'fig')
    
    stc_fname = os.path.join(psd_dir, f'{subj}-{task}-psd-fsaverage')
    fig_fname = os.path.join(fig_dir, f'{subj}-{task}-psd-fsaverage-normative.png')
    try:
        visualize_psd(stc_fname, fig_fname, subj, stc_avg_fname=stc_avg_fname, stc_var_fname=stc_var_fname)
    except OSError:
        continue

