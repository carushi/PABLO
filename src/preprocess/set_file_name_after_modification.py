#!/usr/bin/bash
# File name change after modify_start_point.py

GBM_DIR=tcia_gbm
GLIOMA_DIR=tcia_glioma

mkdir output_${GBM_DIR}
mkdir output_${GLIOMA_DIR}

for DIR in $GBM_DIR $GLIOMA_DIR ;
do
    ls $DIR | xargs -I{} mkdir output_${DIR}/{}
done


# Cleaning

for DIR in $GLIOMA_DIR $GBM_DIR
do
find $DIR / -name "*.cp.nii" | xargs - I{} rm {}
find $DIR / -name "*.mat" | xargs - I{} rm {}
find $DIR / -name "y*.nii" | xargs - I{} rm {}
find $DIR / -name "c*.nii" | xargs - I{} rm {}
done


for DIR in $GLIOMA_DIR $GBM_DIR
do
find $DIR / -name "*Warp.nii.gz" - exec rm {} \
find $DIR / -name "*.mat" - exec rm {} \
done

for DIR in $GLIOMA_DIR $GBM_DIR ;
do
find $DIR -name "*changed.nii" -exec mv {} output_{} \;
find output_$DIR/ -name "*changed.nii" -exec rename 's/_changed//g' {} \;
find output_$DIR/ -name "*.nii" -exec gzip -f {} \;
done

