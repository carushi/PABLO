#!/bin/bash
set -euxo pipefail

# parent directory
IDIR=${1}/
ODIR=${2}/

# dataset name (gbm, gbm_edema, glioma, tcia_gbm, or tcia_glioma)
TYPE=$3
TYPE_DIR=$4
MDIR=$5
# feature computation methods (basic, atlas, or pyradiomics)
METHOD=$6
# start patient id
START=$7
# end patient id
END=$8
# RAW, SPM, or ANTs
BRAIN=$9

RECOMPUTE=${10}
if [ ${RECOMPUTE} = "RECOMPUTE" ]; then
    RENEWAL="--frenewal --renewal "
else
    RENEWAL="--frenewal "
fi
SRENEWAL="--frenewal "
echo $RENEWAL

if [ -z "${TYPE-}" ]; then
    TYPE="gbm"
fi
if [ -z "${METHOD-}" ]; then
    METHOD=
fi
if [ -z "${START-}" ]; then
    START=1
fi
if [ -z "${END-}" ]; then
    END=105
fi
if [ -z "${BRAIN-}" ]; then
    BRAIN="RAW"
fi

if [ "${METHOD}" = "basic" ]; then
	BASIC="--basic"
	PREFIX="b"
elif [ "${METHOD}" = "atlas" ]; then
	BASIC="--atlas"
	PREFIX="a"
else
	BASIC=
	PREFIX="py"
fi

mkdir ${ODIR}Tumor_roi || true
mkdir ${ODIR}Tumor_roi_brain || true # skull-stripped
mkdir ${ODIR}Edema_roi || true
mkdir ${ODIR}Edema_roi_brain || true # skull-stripped
mkdir ${ODIR}No_roi || true


if [ "${TYPE}" = "gbm" ]; then
    OPTIONS="--indir ${IDIR}/${TYPE_DIR} --feature ${PREFIX}gbm $BASIC --start $START --end $END"
    MASK_DIR="${MDIR}/[PATIENT]/"
    if [ "${BRAIN}" == "SPM" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi --prefix 2w --mask ${MASK_DIR}b2w[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi_brain --prefix s2w --mask ${MASK_DIR}b2w[METHOD]_mask.nii ${RENEWAL}
    elif [ "${BRAIN}" == "ANTs" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi --prefix a --mask ${MASK_DIR}ba[METHOD]_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi_brain --prefix ea --mask ${MASK_DIR}ba[METHOD]_mask.nii.gz ${RENEWAL}
    else
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi --prefix '' --mask ${MASK_DIR}$[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi_brain --prefix s --mask ${MASK_DIR}[METHOD]_mask.nii ${RENEWAL}        
    fi
elif [ "${TYPE}" = "gbm_edema" ]; then
    OPTIONS="--indir ${IDIR}/${TYPE_DIR} --feature ${PREFIX}gbm $BASIC --start $START --end $END"
    MASK_DIR="${MDIR}/[PATIENT]/"
    if [ "${BRAIN}" == "SPM" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi --prefix 2w --mask ${MASK_DIR}b2w[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix s2w --mask ${MASK_DIR}b2w[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi --prefix s2w --mask ${MASK_DIR}b2w[METHOD].nii ${RENEWAL}
    elif [ "${BRAIN}" == "ANTs" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi --prefix a --mask ${MASK_DIR}ba[METHOD]_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix ea --mask ${MASK_DIR}ba[METHOD]_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi --prefix ea --mask ${MASK_DIR}ba[METHOD].nii.gz ${RENEWAL}
    else
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi --prefix '' --mask ${MASK_DIR}[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${IDIR}Edema_roi_brain --prefix s --mask ${MASK_DIR}[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi --prefix s --mask ${MASK_DIR}b[METHOD].nii ${RENEWAL}
    fi

elif [ "${TYPE}" = "glioma" ]; then
    OPTIONS="--indir ${IDIR}/${TYPE_DIR} -feature ${PREFIX}glioma $BASIC --start $START --end $END "
    MASK_DIR="${MDIR}/[PATIENT]/"
    if [ "${BRAIN}" == "SPM" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi       --prefix 2w  --mask ${MASK_DIR}b2w[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix s2w --mask ${MASK_DIR}b2w[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_ro        i  --prefix s2w --mask ${MASK_DIR}b2w[METHOD].nii ${RENEWAL}
    elif [ "${BRAIN}" == "ANTs" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi       --prefix a  --mask ${MASK_DIR}ba[METHOD]_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix ea --mask ${MASK_DIR}ba[METHOD]_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi          --prefix ea --mask ${MASK_DIR}ba[METHOD].nii.gz ${RENEWAL}
    else
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi       --prefix '' --mask ${MASK_DIR}[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix s  --mask ${MASK_DIR}[METHOD]_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi          --prefix s  --mask ${MASK_DIR}b[METHOD].nii ${RENEWAL}
    fi

elif [ "${TYPE}" = "tcia_gbm" ]; then
    OPTIONS="--indir ${IDIR}/${TYPE_DIR} --feature ${PREFIX}tcia_gbm $BASIC --start $START --end $END"
    MASK_DIR="${MDIR}/[PATIENT]/"
    if [ "${BRAIN}" == "SPM" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix 2w --mask ${MASK_DIR}b2wGT_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi_brain --prefix 2w --mask ${MASK_DIR}b2wGD_mask.nii.gz ${SRENEWAL}
        if [ "${METHOD}" = "atlas" ]; then
            exit
        fi
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi --prefix 2w --mask ${MASK_DIR}b2w[METHOD].nii.gz ${SRENEWAL}
    elif [ "${BRAIN}" == "ANTs" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix a --mask ${MASK_DIR}baGT_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi_brain --prefix a --mask ${MASK_DIR}baGD_mask.nii.gz ${SRENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi $BASIC   --prefix a --mask ${MASK_DIR}ba[METHOD].nii.gz ${SRENEWAL}
    else
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix '' --mask ${MASK_DIR}GT_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Tumor_roi_brain --prefix '' --mask ${MASK_DIR}GD_mask.nii.gz ${SRENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi          --prefix '' --mask ${MASK_DIR}b[METHOD].nii.gz ${SRENEWAL}
    fi
elif [ "${TYPE}" = "tcia_glioma" ]; then
    OPTIONS="--indir ${IDIR}/${TYPE_DIR} --feature ${PREFIX}tcia_glioma $BASIC --start $START --end $END"
    MASK_DIR="${MDIR}/[PATIENT]/"
    if [ "${BRAIN}" == "SPM" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix 2w --mask ${MASK_DIR}b2wGT_mask.nii.gz ${RENEWAL}
        if [ "${METHOD}" = "atlas" ]; then
            exit
        fi
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi --prefix 2w --mask ${MASK_DIR}b2w[METHOD].nii.gz ${SRENEWAL}
    elif [ "${BRAIN}" == "ANTs" ]; then
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix a --mask ${MASK_DIR}baGT_mask.nii ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi          --prefix a --mask ${MASK_DIR}ba[METHOD].nii.gz ${SRENEWAL}
    else
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}Edema_roi_brain --prefix '' --mask ${MASK_DIR}GT_mask.nii.gz ${RENEWAL}
        python compute_normed_features.py $OPTIONS --outdir ${ODIR}No_roi          --prefix '' --mask ${MASK_DIR}b[METHOD].nii.gz ${SRENEWAL}
    fi
else
    # print the number of files under each patient directory (4-digit id)
	ls | sed 's/[0-9]\{4\}/0000/g' | sort | uniq | grep GD | \
	while read file; do
	find ./ -name ${file/0000/*} | awk '{if (NR==1) {print $0}; str=$0} END{print NR; system("ls -lht " $str)}'
	done
fi








