#!/bin/bash
#SBATCH --time=0-02:00:00    # 2 hours
#SBATCH --mem-per-cpu=16000    # 16000MB of memory
#SBATCH --array=0-5
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

if  [ -z "$MRI_DIR" ]
then
  echo "MRI_DIR not set, exiting"
  exit 1
fi

if  [ -z "$SUBJECTS_DIR" ]
then
  echo "SUBJECTS_DIR not set, exiting"
  exit 1
fi

echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
temp=$SUBJECTS_DIR

module load matlab/r2017b
module load spm/12.2019
#module load mne
module load freesurfer

export SUBJECTS_DIR=$temp

srun matlab -nosplash -nodesktop -r "try; cd $PWD; disp(pwd); calcFids($SLURM_ARRAY_TASK_ID); catch e; disp(e); end; exit";


