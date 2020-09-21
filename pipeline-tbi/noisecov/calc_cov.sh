#!/bin/bash

module load anaconda3
source activate neuroimaging

cwd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"

if  [ -z "$ER_DIR" ]
then
  echo "ER_DIR not set, exiting"
  exit 1
fi

if  [ -z "$OUTPUT_DIR" ]
then
  echo "OUTPUT_DIR not set, exiting"
  exit 1
fi

cd $ER_DIR || exit 1

for f in $(find "$ER_DIR" -name "*.fif"); do
  python $cwd/noisecov.py $f $OUTPUT_DIR
done
