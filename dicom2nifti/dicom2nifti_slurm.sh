#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --output=./slurm_logs/slurm-%A.out

if  [ -z "$MRI_DIR" ]
then
  echo "MRI_DIR not set, exiting"
  exit 1
fi

module load matlab/r2017b
module load spm/12.2019

srun matlab -nosplash -nodesktop -r "try; cd $PWD; disp(pwd); dicom2nifti(\"$MRI_DIR\"); catch e; disp(e); disp(e.stack); end; exit";

