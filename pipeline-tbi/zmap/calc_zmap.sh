#!/bin/bash

SUBJECTS_DIR=/scratch/work/italinv1/tbi/mri_recons
OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/processed
AVERAGES_DIR=/scratch/nbe/tbi-meg/veera/averages/camcan

echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

cd $SUBJECTS_DIR || exit 1

dirnames=(*/)

COHORTS=$1
TASK=EC

for d in "${dirnames[@]}"
do
  sub=${d%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    python /scratch/nbe/tbi-meg/veera/pipeline/zmap/calc_zmap.py $sub $TASK $OUTPUT_DIR $SUBJECTS_DIR $AVERAGES_DIR $COHORTS
  fi
done
