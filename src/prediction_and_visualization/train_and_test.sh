#!/usr/bin/bash
set -euxo pipefail

ANN_FEATURE="Age1_n,Sex,KPS_n"
# Edema_roi, Tumor_roi_brain, No_roi, etcetc.
TYPE=$1
# Test data
DIR=$2
# Training data
TDIR=$3
# Prediction, prediction with update, update_only
CLASSIFICATION=$4
# Regexp files
RDIR=$6
# Annotation table files
ANNOTATION_FILE=$7
# all, annotation, image
FEATURE_SET=$8


if [ -z "${TYPE-}" ]; then
    TYPE="Edema_roi"
fi
if [ -z "${METHOD-}" ]; then
    METHOD="1"
fi

if [ "${TYPE}" = "Tumor_roi" ] || [ "${TYPE}" = "Tumor_roi_brain" ]; then
    GBM=" --gbm "
    PATIENT="../../ann/gbm_id_list.txt"
    VPATFILE="../../ann/validation_gbm_id.txt"
else
    GBM=
    PATIENT="../../ann/patient_id_list.txt"
    VPATFILE="../../ann/validation_id_list.txt"
fi

FPREFIX=
RPREFIX=
SCRIPT=classification_feature_tables.py

for METHOD in {0..3}; do
    if [ "${METHOD}" = "0" ]; then
        # Outlier filtering
        NORM="[ID]__[NORM]_[METHOD]_k"
        VNORM="[ID]__[NORM]_[METHOD]_k"
        PREFIX=""
        PREFLAG=""
        DPREFIX=""
        POST=''
    elif [ "${METHOD}" = "1" ]; then
        # Outlier + post dimension-reduction
        NORM="[ID]__[NORM]_[METHOD]_k"
        VNORM="[ID]__[NORM]_[METHOD]_k"
        PREFIX="_post"
        PREFLAG=""
        DPREFIX=""
        POST=' --post '
    elif [ "${METHOD}" = "2" ]; then
        # Z-score + post dimension-reduction
        NORM="[ID]__[NORM]_[METHOD]_z"
        VNORM="[ID]__[NORM]_[METHOD]_z"
        PREFIX="_zscore"
        DPREFIX="_zscore"
        FPREFIX='_z'
        RPREFIX="_zscore"
        PREFLAG="--output ${PREFIX} "
        POST=' --post '
    elif [ "${METHOD}" = "3" ]; then
        # Z-score
        NORM="[ID]__[NORM]_[METHOD]_z"
        VNORM="[ID]__[NORM]_[METHOD]_z"
        PREFIX="_rzscore"
        DPREFIX="_rzscore"
        FPREFIX='_z'
        RPREFIX='_zscore'
        PREFLAG="--output _zscore "
        POST=''
    else
        echo
    fi

    VPATIENT=" --regex ${RDIR}prefix_regex${FPREFIX}.txt,${RDIR}prefix_regex_valid${FPREFIX}.txt --dir ${DIR},${TDIR} --patient ${PATIENT},${VPATFILE} --normalize_regexp $NORM,$VNORM "

    if [ -z ${FEATURE_SET-} || "${FEATURE_SET}" = "all" ]; then
        BASE_OPTION=" --ann $ANN --ann_feature $ANN_FEATURE $PREFLAG $GBM $VPATIENT "
    elif [ "${FEATURE_SET}" = "annotation" ]; then
        BASE_OPTION=" --ann $ANN --ann_feature $ANN_FEATURE $PREFLAG $GBM $VPATIENT --ann_pred "
    else # image only
        BASE_OPTION=" --ann $ANN $PREFLAG $GBM $VPATIENT "
    fi

    if [ "${METHOD}" = "0" ] || [ "${METHOD}" = "2" ]; then
        if [ "$CLASSIFICATION" = "update" ]; then
            echo;
            : "Update dataset" && {
                echo;
                python $SCRIPT $BASE_OPTION  --update  > parse_out_filt$RPREFIX.txt
                python $SCRIPT $BASE_OPTION --clust    > parse_out_clust$DPREFIX.txt
            }
        fi
        if [ "$CLASSIFICATION" = "rawupdate" ] || [ "$CLASSIFICATION" = "update" ]; then
            : "Raw" && {
                echo;
                if [ "${GBM}" = " --gbm " ]; then
                    MAX=2
                else
                    MAX=16
                fi
                for i in ` seq 0 ${MAX} `
                do
                    echo "python ${SCRIPT} $BASE_OPTION  \\" > auc_script${FPREFIX}.sh
                    echo " --raw --sp ${i} --auc > ${i}_raw${RPREFIX}.txt " >> auc_script${FPREFIX}.sh
                    bash auc_script${FPREFIX}.sh
                    echo auc
                done
                grep -h Raw_ [0-9]*_raw$RPREFIX.txt | grep -v "_compressed" |sed -e 's/Group /Group_/g' -e 's/ GBM/_GBM/g' -e 's/ LGG/_LGG/g' > raw_auc$RPREFIX.txt
            }
        fi
    fi

    if [ "$CLASSIFICATION" = "rawupdate" ]; then
        exit
    fi

    : "NMF" && {
        echo;
    python $SCRIPT $BASE_OPTION --auc $POST --nmf > nmf_auc_all$PREFIX.txt 2>&1
    grep " NMF_" nmf_auc_all$PREFIX.txt | grep -v "_compressed" | gsed -e 's/Group /Group_/g' -e 's/ GBM/_GBM/g' -e 's/ LGG/_LGG/g' > nmf_auc$PREFIX.txt
    }

    : "PCA" && {
        echo;
    python $SCRIPT $BASE_OPTION --auc  $POST --pca > pca_auc_all$PREFIX.txt 2>&1
    grep " PCA_" pca_auc_all$PREFIX.txt | grep -v "_compressed" |sed -e 's/Group /Group_/g' -e 's/ GBM/_GBM/g' -e 's/ LGG/_LGG/g' > pca_auc$PREFIX.txt
    }

    : "MDS" && {
        echo;
    python $SCRIPT $BASE_OPTION --auc $POST --mds > mds_auc_all$PREFIX.txt 2>&1
    grep " MDS_" mds_auc_all$PREFIX.txt | grep -v "_compressed" |sed -e 's/Group /Group_/g' -e 's/ GBM/_GBM/g' -e 's/ LGG/_LGG/g' > mds_auc$PREFIX.txt
    }

done