#!/bin/bash


OUTPUT_DIR=/run/user/1578013/gvfs/smb-share:server=data.triton.aalto.fi,share=scratch/nbe/tbi-meg/veera/processed

task=$1

for d in "$OUTPUT_DIR"/0*
do
  sub=${d##*/}
  sub=${sub%/}
  report=${d}/ica/${sub}-${task}-ica-report.html
  echo "${report}"
  if [ -e "${report}" ]
  then
    firefox --new-tab --url "${report}"
  fi
done
