#!/bin/bash

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

GROUP=$1

if  [ -z "$GROUP" ]
then
  GROUP=total
fi

python /scratch/nbe/tbi-meg/veera/pipeline/averages/get_avg.py "${GROUP}"
