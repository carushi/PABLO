#!/usr/bin/bash
set -euo pipefail

# Set your ANTs source directory properly
source ../dir_list/directory_list_ants.txt
ANTSCRIPT=${ANTS_DIR}/Scripts/

if [ "${1}" = "glioma" ]; then
MASK="GT"
elif [ "${1}" = "gbm" ]; then
MASK="GD"
else
MASK=""
fi

echo $INPUT
echo $OUTPUT

GREP=${2:-}
if [ -z ${GREP} ]; then
    GREP=' -e ^[0-9]\{4\}$'
elif [ "${GREP}" == "-1"]; then
    GREP=\^0
else
    GREP=$2
fi

cd ${INPUT}
echo $GREP

ls  | grep $GREP | \
while read id
do
    echo $id
    mkdir $OUTPUT/$id || true
    cd $id
    for norm in SPM RAW ANTS;  do
        for met in z k; do # z=z-score-based, k=outlier filtering
            NORM=$met
            echo $norm    
            for file in ` echo GD GT T1 T2 FLAIR | xargs -n 1 -I{} ls {}.nii.gz | cut -f1 -d"." `
            do
                if [ "$norm" = "ANTS" ]; then
                    FILT="a{}.nii.gz"
                    TAIL=".nii.gz"
                elif [ "$norm" = "SPM" ]; then
                    FILT="2w{}.nii.gz"
                    TAIL=".nii.gz"
                else
                    FILT="{}.nii.gz"
                    TAIL=".nii.gz"
                fi
                if [ "${MASK}" == "" ]; then
                    mask=${file}
                else
                    mask=${MASK}
                fi
                if [ "$norm" = "ANTS" ]; then
                    ExtractRegionFromImageByMask 3 ${NORM}${file}.nii.gz $OUTPUT/$id/r${NORM}${file}.nii.gz ba${mask}_mask.nii.gz 1 2
                    ExtractRegionFromImageByMask 3 ${NORM}e${file}.nii.gz $OUTPUT/$id/r${NORM}e${file}.nii.gz bea${mask}_mask.nii.gz 1 2
                elif [ "$norm" = "SPM" ]; then
                    ExtractRegionFromImageByMask 3 ${NORM}${file}.nii.gz $OUTPUT/$id/r${NORM}${file}.nii.gz b2w${mask}_mask.nii 1 2
                    ExtractRegionFromImageByMask 3 ${NORM}s${file}.nii.gz $OUTPUT/$id/r${NORM}s${file}.nii.gz bs2w${mask}_mask.nii 1 2
                else
                    ExtractRegionFromImageByMask 3 ${NORM}${file}.nii.gz $OUTPUT/$id/r${NORM}${file}.nii.gz ${mask}_mask.nii 1 1
                    ExtractRegionFromImageByMask 3 ${NORM}s${file}.nii.gz $OUTPUT/$id/r${NORM}s${file}.nii.gz ${mask}_mask.nii 1 1
                fi
            done
            if [ "$SSHDIR" != "" ]; then
                if [ "$norm" = "ANTS" ]; then
                    echo ANTS
                    scp $OUTPUT/$id/r${NORM}${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp $OUTPUT/$id/r${NORM}e${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp ${NORM}${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp ${NORM}e${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                elif [ "$norm" = "SPM" ]; then
                    echo SPM
                    scp $OUTPUT/$id/r${NORM}${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp $OUTPUT/$id/r${NORM}s${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp ${NORM}${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp ${NORM}s${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                else
                    echo RAW
                    scp $OUTPUT/$id/r${NORM}${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp $OUTPUT/$id/r${NORM}s${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp ${NORM}${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                    scp s${file}.nii.gz ${SCP}:${SSHDIR}/${id}/
                fi
                sleep 15
            fi
        done
    done
    cd ../
done