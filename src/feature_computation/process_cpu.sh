#!/bin/bash

#PBS --group=g-rwbc-oil
#PBS -o logs/job_out.%s
#PBS -e logs/job_err.%s
#PBS -q cq
#PBS -l cpunum_job=5
#PBS -l elapstim_req=42:00:00
#PBS -t 0-0:1
###PBS -t 0-4:1

set -x
date

cd $PBS_O_WORKDIR

pwd
echo ${PWD}

echo $PBS_O_ARG1



### change following lines for keras environment
source activate py36
module switch cuda/8.0 cuda/8.0.61+cudnn-6.0.21+nccl-1.3.4-1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=${PWD}:$PYTHONPATH
###




echo $HOSTNAME
echo $GPUID
echo $SAMPLEID

echo python process_nii.py -list $ARG_L -out $ARG_O

python process_nii.py -list $ARG_L -out $ARG_O


date
set +x
