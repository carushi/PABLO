#!/usr/bin/bash
set -euo pipefail

# Set your ANTs source directory properly
source ../dir_list/directory_list_ants.txt
# brain type '', 2w, a
PREFIX=$1
# compressed or not '', .gz
SUFFIX='.gz'

GREP=${2:-}
if [ -z "$GREP" ]; then
    GREP='^[0-9]\{4\}$'
    echo $GREP
elif [ "${GREP}" = "-1" ]; then
    GREP=\^0
fi
ls ${INPUT} | grep $GREP | \
while read id
do
    echo $id
    mkdir $OUTPUT/$id || true
    for full_file in ` echo GD GT T1 T2 FLAIR | xargs -n 1 -I{} ls ${INPUT}/$id/{}.nii${SUFFIX} | awk '{print gensub(".*/", "", $0)}' 2>/dev/null `
    do
        file=${full_file/.nii${SUFFIX}/}
        if [ -f "$OUTPUT/$id/s${PREFIX}${file}.nii.gz" ]; then
            continue
        fi
        echo ${INPUT}/$id/${PREFIX}${file}.nii${SUFFIX}
        standard_space_roi ${INPUT}/$id/${file}.nii${SUFFIX} $OUTPUT/$id/s${PREFIX}${file}.nii${SUFFIX} -b
    done
done
