#!/usr/bin/bash
set -euo pipefail

# Set your ANTs source directory properly
source ../dir_list/directory_list_ants.txt
ANTSCRIPT=${ANTS_DIR}/Scripts/


# Convert the space of ANTs template

TEMPLATE_FILE="${ANTS_DIR}/template/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii"
MASK_FILE="${ANTS_DIR}/template/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumMask.nii"

TEMPLATE_RSL="${ANTS_DIR}/template/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_rsl_swap.nii.gz"
MASK_RSL="${ANTS_DIR}/template/MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumMask_rsl_swap.nii.gz"

for FILE in $TEMPLATE_FILE $MASK_FILE
do
    OUT_FILE="${FILE/.nii/_rsl.nii.gz}"
    if [ -f "${OUT_FILE/_rsl/_rsl_swap}" ]; then
    continue
    fi
    echo $FILE $OUT_FILE
    fslswapdim $FILE -x y z $OUT_FILE
    fslorient -swaporient $OUT_FILE
    rename -f "s/_rsl/_rsl_swap/g" $OUT_FILE
done

two_masks=false
if [ "${1}" = "glioma" ]; then
MASK="GT"
elif [ "${1}" = "gbm" ]; then
MASK="GD"
else
# Training data from TCIA
# Repository contains GD and GT masks
two_masks=true 
MASK="GD"
fi

SUFFIX=".nii.gz"
MSUFFIX=".nii.gz"

echo $INPUT
echo $OUTPUT

GREP=${2:-}
if [ -z ${GREP} ]; then
    GREP='^[0-9]\{4\}$'
elif [ "${GREP}" = "-1"]; then
    GREP=\^0
else
    GREP=$2
fi

exit
ls ${INPUT} | grep $GREP | \
while read id
do
    echo $id
    mkdir $OUTPUT/$id || true
    for file in ` echo GD GT T1 T2 FLAIR | xargs -n 1 -I{} ls ${INPUT}/$id/{}${SUFFIX} | awk -F"/" '{print $NF}'| cut -f1 -d"." `
    do
        echo $file
        # : "Skip a file if you need" && {
        #     if [ -f "$OUTPUT/$id/a${file}.nii.gz" ]; then
        #         continue
        #     fi
        # }
        echo $id $file
        : "Homography-based projection" && {
            if [ -f "$OUTPUT/$id/a${file}.nii.gz" ]; then
                echo 'already computed'
            else
                # ${ANTSPATH}/antsApplyTransforms -d 3 -i $id/GD_mask.nii.gz -o $OUTPUT/$id/a${file}_mask.nii.gz -r $TEMPLATE -t $OUTPUT/$id/a${file}1Warp.nii.gz -t $OUTPUT/$id/a${file}0GenericAffine.mat
                bash ${ANTSCRIPT}/antsRegistrationSyNQuick.sh -n 5 -d 3 -f $TEMPLATE_RSL -m "${INPUT}/$id/${file}${SUFFIX}" -o "$OUTPUT/$id/a${file}"
                # n = thread number
                mv $OUTPUT/$id/a${file}Warped.nii.gz $OUTPUT/$id/a${file}.nii.gz
            fi
            if [ -f "$OUTPUT/$id/ea${file}.nii.gz" ]; then
                echo 'already computed'
            else
                ${ANTSPATH}/ImageMath 3 ${OUTPUT}/$id/ea${file}.nii.gz m $OUTPUT/$id/a${file}.nii.gz $MASK_RSL
            fi
            if $two_masks; then
                if [ "${file}" = "GD" ]; then
                    echo $id/
                    ${ANTSPATH}/antsApplyTransforms -d 3 -i ${INPUT}/$id/${MASK}_mask${MSUFFIX} -o ${OUTPUT}/$id/a${file}_mask.nii.gz -r $TEMPLATE_RSL -t $OUTPUT/$id/a${file}1Warp.nii.gz -t $OUTPUT/$id/a${file}0GenericAffine.mat
                    ${ANTSPATH}/antsApplyTransforms -d 3 -i ${INPUT}/$id/GT_mask${MSUFFIX} -o ${OUTPUT}/$id/aGT_mask.nii.gz -r $TEMPLATE_RSL -t $OUTPUT/$id/a${file}1Warp.nii.gz -t $OUTPUT/$id/a${file}0GenericAffine.mat
                fi
            else
                ${ANTSPATH}/antsApplyTransforms -d 3 -i ${INPUT}/$id/${file}_mask${MSUFFIX} -o ${OUTPUT}/$id/a${file}_mask.nii.gz -r $TEMPLATE_RSL -t $OUTPUT/$id/a${file}1Warp.nii.gz -t $OUTPUT/$id/a${file}0GenericAffine.mat
            fi
        }
        : "Remove unnecessarry files" && {
            rm ${OUTPUT}/$id/a${file}1InverseWarp.nii.gz || true
            rm ${OUTPUT}/$id/a${file}_lap.nii.gz || true
            rm ${OUTPUT}/$id/${file}_prob?.nii.gz || true
            rm ${OUTPUT}/$id/a${file}_seg_mask.nii.gz || true
            rm ${OUTPUT}/$id/a${file}InverseWarped.nii.gz || true
            # rm $OUTPUT/$id/a${file}0GenericAffine.mat || true
            # rm $OUTPUT/$id/a${file}1Warp.nii.gz || true
        }
    done
done
