#!/bin/bash

OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/processed

echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

cd "$OUTPUT_DIR" || exit 1

dirnames=(*/)

for d in "${dirnames[@]}"
do
  sub=${d%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    srun python /scratch/nbe/tbi-meg/veera/pipeline/window/create_windowed_data.py ${sub} EC ${OUTPUT_DIR}
  fi
done