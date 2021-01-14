#!/usr/bin/bash
# For setting the same orientation (radiological to neurological)
# ANTs images

GBM_DIR=tcia_gbm
GLIOMA_DIR=tcia_glioma

for DIR in $GBM_DIR $GLIOMA_DIR ;
do
find $DIR -name "*.nii" | xargs -I{} echo fslswapdim {} -x y z {}.gz | bash
find $DIR -name "*.nii.gz" | xargs -I{} echo rename \'s/_rsl/_rsl_swap/g\' {} 
find $DIR -name "*_swap.nii.gz" | xargs -I{} echo fslorient -swaporient {} | bash
find $DIR -name "*_swap.nii.gz" | xargs -I{} echo rename 's/_rsl_swap//g' {} | bash
find $DIR -name "*.nii.gz" | xargs -I{} echo gunzip -k {} | bash
done

for DIR in $GLIOMA_DIR $GBM_DIR ;
do
echo $DIR
find $DIR/ -name "*.nii" | grep -v changed | xargs -I{} cp {} {}.cp.nii
find $DIR/ -name "*.cp.nii" | grep -v changed | xargs -I{} fslswapdim {} -x y z {}.swap
find $DIR/ -name "*.cp.nii.swap.nii.gz" | grep -v changed | xargs -I{} fslorient -swaporient {}
done

