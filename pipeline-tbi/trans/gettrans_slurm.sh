#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --mem-per-cpu=8000
#SBATCH --array=0-4
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

SUBJECTS_DIR=/scratch/work/italinv1/tbi/mri_recons
OUTPUT_DIR=/scratch/nbe/tbi-meg/veera/processed

echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
echo "OUTPUT_DIR set as $OUTPUT_DIR"

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

cd $SUBJECTS_DIR || exit 1

CHUNKSIZE=10
n=$SLURM_ARRAY_TASK_ID
indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`

dirnames=(*/)
for i in $indexes
do
  sub=${dirnames[$i]}
  sub=${sub%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    srun xvfb-run -a python /scratch/nbe/tbi-meg/veera/pipeline/trans/calc_trans.py ${sub} ${SUBJECTS_DIR} ${OUTPUT_DIR}
  fi
done
