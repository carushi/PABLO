# Mask conversion and tumor extraction

## 1. Binarization

* Edit mask_directory_list.txt
  * bash aligned2mask_nii.sh output_tcia_gbm

## 2. Tumor extraction
* Edit directory_list_ants.txt 
* Change OUTPUT directory for each mask type
  * bash mask_specific_file_dir.sh gbm # tissue
  * bash mask_specific_file_dir.sh glioma # edema
  * bash mask_specific_file_dir.sh local # use different masks for each sequence
