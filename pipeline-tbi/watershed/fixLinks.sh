#!/bin/bash

if  [ -z "$SUBJECTS_DIR" ]
then
  echo "SUBJECTS_DIR not set, exiting"
  exit 1
fi

cwd="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"

for d in $SUBJECTS_DIR/*/
do
    cd $d/bem/
    echo $d
    $cwd/fixLink.sh
    
done
