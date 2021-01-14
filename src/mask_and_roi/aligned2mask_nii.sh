#!/usr/bin/bash
set -euo pipefail

INPUT="mask_directory_list.txt"
DIR=`cat $INPUT`
echo $DIR

DATASET=$1
STRIPPED_FLAG='tcia'
# STRIPPED_FLAG='' keyword to detct the directory whose data is already skull-stripped
TARGET=${2:-}
if [ -z ${TARGET} ]; then
TARGET=" -e '^.*$' "
fi
echo $TARGET
echo $DATASET
echo $STRIPPED_FLAG

echo "find ${DIR}/${DATASET}/ -type d -mindepth 1 | grep $TARGET" | bash | \
 xargs -I{} echo "echo {} && python aligned2mask_nii.py {} ${STRIPPED_FLAG} " |  bash 
echo "find ${DIR}/${DATASET}/ -type d -mindepth 1 | grep $TARGET" | bash | \
 xargs -I{} echo "python aligned2mask_nii.py --mask {} " |  bash

