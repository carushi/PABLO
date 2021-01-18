#!/bin/bash

IDIR=/path/to/directory/data/nifti/
ODIR=/path/to/directory/data/features/
MDIR=/path/to/directory/data/nifti/

RECOMPUTE="\' \'"

TYPE="gbm"
TYPE_DIR="local_gbm"
for METHOD in basic atlas pyradiomics; do
    START=1
    END=102
    for BRAIN in RAW SPM ANTs; do
        bash compute_normed_features.sh $IDIR $ODIR/${TYPE} $TYPE $TYPE_DIR $MDIR/$TYPE_DIR $METHOD  $START $END $BRAIN $RECOMPUTE
    done
done


TYPE="gbm_edema"
TYPE_DIR="local_gbm_edema"
for METHOD in basic atlas pyradiomics; do
    START=1
    END=102
    for BRAIN in RAW SPM ANTs; do
        bash compute_normed_features.sh $IDIR $ODIR/${TYPE} $TYPE $TYPE_DIR $MDIR/local_gbm $METHOD  $START $END $BRAIN $RECOMPUTE
    done
done


TYPE="glioma"
TYPE_DIR="local_glioma"
for METHOD in basic atlas pyradiomics; do
    START=1
    END=82
    for BRAIN in RAW SPM ANTs; do
        bash compute_normed_features.sh $IDIR $ODIR/${TYPE} $TYPE $TYPE_DIR $MDIR/$TYPE_DIR $METHOD  $START $END $BRAIN $RECOMPUTE
    done 
done

TYPE="tcia_gbm"
TYPE_DIR="output_tcia_gbm"
for METHOD in basic atlas pyradiomics; do
    START=1
    END=150
    for BRAIN in RAW SPM ANTs ; do
        bash compute_normed_features.sh $IDIR $ODIR/${TYPE} $TYPE $TYPE_DIR $MDIR/$TYPE_DIR $METHOD  $START $END $BRAIN $RECOMPUTE
    done
done

TYPE="tcia_glioma"
TYPE_DIR="output_tcia_glioma"
for METHOD in basic atlas pyradiomics; do
    START=1
    END=150
    for BRAIN in RAW SPM ANTs; do
        bash compute_normed_features.sh $IDIR $ODIR/${TYPE} $TYPE $TYPE_DIR $MDIR/$TYPE_DIR $METHOD  $START $END $BRAIN $RECOMPUTE
    done
done
