#!/bin/bash

module load anaconda3
source activate neuroimaging

OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/processed

echo "OUTPUT_DIR set as $OUTPUT_DIR"

cd "$OUTPUT_DIR" || exit 1

dirnames=(*/)

for d in "${dirnames[@]}"
do
  sub=${d%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    srun python /scratch/nbe/tbi-meg/veera/pipeline/noisecov/noisecov.py ${sub} ${OUTPUT_DIR}
  fi
done

