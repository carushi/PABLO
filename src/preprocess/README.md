# Preprocessing

## 1. Set a correct path

* Edit directory_list.txt
  * Put training-testing GBM-LGG data in a different directory
    * data/nifti/original_tcia_gbm, data/nifti/original_tcia_glioma
    * data/dicom/local_gbm, data/dicom/local_glioma

## 2. Prepare datasets

* Dataset format

  * From Dicom (local dataset)
    * Convert to Nifti
    * dicom_to_nifti.m: patient id is 0001, 0002, etcetc
    * dicom_to_nifti_local.m: any directory name
    * Public dataset from NIFTI files

  * From NIFTI (public dataset)
    * Copy TCIA files to a new directory with the specific name format
      * python convert_TCIA_dataset.py from to
      * python convert_TCIA_dataset.py original_tcia_glioma tcia_glioma
      * python convert_TCIA_dataset.py original_tcia_gbm tcia_gbm
    *  Convert a start point (S-form and Q-form) for SPM conversion
      * python modify_start_point.py
      * apply_segmentation_tcga.m

## 3. Spatial normalization

* 