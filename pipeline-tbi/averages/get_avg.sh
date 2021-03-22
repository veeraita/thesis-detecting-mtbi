#!/bin/bash

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

COHORTS=$1
GROUP=camcan

python /scratch/nbe/tbi-meg/veera/pipeline/averages/get_avg.py "${GROUP}" "${COHORTS}"
