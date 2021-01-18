#!/usr/bin/bash

DIR=$1

if [[ -d $DIR ]]; then
    for file in ` ls ${DIR}/*.dat `; do
        ARG_L=${file}
        ARG_O=${file/.dat/}_cpu.res
        python process_nii.py -list $ARG_L -out $ARG_O
    done
elif [[ -f $DIR ]]; then
    file=$DIR
    ARG_L=${file}
    ARG_O=${file/.dat/}_cpu.res
    python process_nii.py -list $ARG_L -out $ARG_O
fi
