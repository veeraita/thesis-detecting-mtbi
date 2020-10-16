#!/bin/bash

if  [ -z "$OUTPUT_DIR" ]
then
  echo "OUTPUT_DIR not set, exiting"
  exit 1
fi

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

export SUBJECTS_DIR=/scratch/work/italinv1/tbi/mri_recons

echo "OUTPUT_DIR set as $OUTPUT_DIR"
echo "SUBJECTS_DIR set as $SUBJECTS_DIR"

GROUP=$1

if  [ -z "$GROUP" ]
then
  GROUP=total
fi

AVG_OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/averages
CAMCAN_DIR=/scratch/nbe/restmeg/veera/processed

python /scratch/nbe/tbi-meg/veera/pipeline/averages/get_avg.py "${OUTPUT_DIR}" "${SUBJECTS_DIR}" "${AVG_OUTPUT_DIR}" "${GROUP}" "${CAMCAN_DIR}"
