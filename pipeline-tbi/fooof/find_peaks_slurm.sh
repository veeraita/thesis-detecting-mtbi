#!/bin/bash
#SBATCH --time=0-01:30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --array=0-73
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

ml purge
module load teflon
ml anaconda3
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate mne

TBI_DIR=/scratch/nbe/tbi-meg/veera/processed/
CAMCAN_DIR=/scratch/nbe/restmeg/veera/processed/

CHUNKSIZE=10
n=$SLURM_ARRAY_TASK_ID
indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`


cd $TBI_DIR || exit 1

tbi_dirs=(*/)

cd $CAMCAN_DIR || exit 1

camcan_dirs=(sub*/)

dirnames=( "${tbi_dirs[@]}" "${camcan_dirs[@]}" )

for i in $indexes
do
  sub=${dirnames[$i]}
  sub=${sub%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    srun python /scratch/nbe/tbi-meg/veera/pipeline/fooof/find_peaks.py "${sub}"
  fi
done
