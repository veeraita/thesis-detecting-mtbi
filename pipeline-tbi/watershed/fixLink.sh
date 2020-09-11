#!/bin/bash

links=($(readlink *.surf))

echo $(pwd)
echo "${links[@]}"
if [[ ${#links[@]} == 0 ]]
then
    for l in ./watershed/sub-*
    do
	surf=${l#*sub*_}
	surf=${surf%%_surface}.surf
	tgt=${l#*bem/}
	echo $tgt
	echo $surf
	ln -s $tgt $surf
    done
    
else

    for l in "${links[@]}"
    do
	varL=${l#./}
	varL=${varL%%/*}
	echo $varL
	if [ "$varL" != "watershed" ]
	then
	    surf=${l#*_}
	    surf=${surf%_surface}.surf
	    echo $surf
	    tgt=./${l#*bem/}
	    echo $tgt
	    rm $surf
	    ln -s $tgt $surf
	fi

    done
fi
