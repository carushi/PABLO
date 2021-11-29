# Requirement
* TPM.nii
 * Download from [https://github.com/spm/spm12](https://github.com/spm/spm12)
 * tpm/TPM.nii
 * Place at ann folder
* priors[1-6].nii.gz
 * Download from [http://stnava.github.io/ANTs/](http://stnava.github.io/ANTs)
 * ANTs/template/MICCAI2012-Multi-Atlas-Challenge-Data/Priors2/priors[NUM].nii.gz"
 * Place at ann folder 

# Training and testing of classifiers

* Train and test classifiers
  * train_and_test.sh
    * example
      * bash TYPE TEST_DIR TRAINING_DIR CLASSIFICATION RDIR ANNOTATION_FILE FEATURE_SET train_and_test.sh
      * bash gbm ../../data/feature/local_gbm/Edema_roi_brain ../../data/feature/tcia_gbm/Edema_roi_brain prediction ../../data/ann/RAW/ ../../data/ann/[annotation_file (id + genotype)] all train_and_test.sh # use all features
      * bash gbm ../../data/feature/local_gbm/Edema_roi_brain ../../data/feature/tcia_gbm/Edema_roi_brain prediction ../../data/ann/RAW/ ../../data/ann/[annotation_file (id + genotype)] image train_and_test.sh # use only image features
      * bash  ../../data/feature/local_glioma/Edema_roi_brain ../../data/feature/tcia_glioma/Edema_roi_brain prediction ../../data/ann/RAW/ ../../data/ann/[annotation_file (id + genotype)] image train_and_test.sh # use only image features
    * to train classifiers for both gbm and glioma
      * copy all feature files under local_gbm and local_glioma to the same directory
      * copy all feature files under tcia_gbm and tcia_glioma to the same directory
      * run train_and_test.sh to these directories
  * AUC output
    * raw_auc['', '_post', '_zscore', '_rzscore'].txt
    * nmf_auc['', '_post', '_zscore', '_rzscore'].txt
    * pca_auc['', '_post', '_zscore', '_rzscore'].txt
    * mds_auc['', '_post', '_zscore', '_rzscore'].txt
* Regexp of feature file names
  * data/ann/RAW, SPM, ANTs
  * example files for Edema_roi_brain (ROI: edema region, skull-stripped)