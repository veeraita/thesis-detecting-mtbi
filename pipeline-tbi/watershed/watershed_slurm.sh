#!/bin/bash
#SBATCH --time=0-00:60:00
#SBATCH --mem-per-cpu=8000    # 8000MB of memory
#SBATCH --array=0-5
#SBATCH --output=./slurm_logs/slurm-%A_%a.out

if  [ -z "$SUBJECTS_DIR" ]
then
  echo "SUBJECTS_DIR not set, exiting"
  exit 1
fi

echo "SUBJECTS_DIR set as $SUBJECTS_DIR"
temp=$SUBJECTS_DIR

module load teflon
module load mne
source $MNE_ROOT/bin/mne_setup_sh

module load freesurfer

# re-set subjects dir after loading modules
export SUBJECTS_DIR=$temp

CHUNKSIZE=10
n=$SLURM_ARRAY_TASK_ID
indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`

cd $SUBJECTS_DIR || exit 1
dirnames=(*/)
for i in $indexes
do
  sub=${dirnames[$i]}
  sub=${sub%?}
  if [ -n "$sub" ]
  then
    echo "${sub}"
    srun mne_watershed_bem --subject "${sub}" --overwrite
  fi
done

