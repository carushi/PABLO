# Feature vector computation

## 1. Pyradiomics and basic features
  * run_feature_computation.sh
    * to run compute_normed_features.sh
    * edit IDIR (input), ODIR (output), and MDIR (mask dir)
  * compute_normed_features.sh
    * to run compute_normed_features.py
    * example:
      * python compute_normed_features.py --indir input_directory --outdir output_directory --feature prefix_for_feature_label (--basic) --start start_id --end end_id --prefix prefix_for_nifti_image([s/2w/ea]) --mask mask_file_regexp(e.g., ba[METHOD]_mask.nii.gz) (--frenewal to recompute feature files) (--renewal to recompute normed image files)
  * If the file naming system is different, directly run a python script.

## 2. Inception Resnet v2
* inception_resnetv2_changed.h5
  * Keras model
* Reference
  * [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
* Make the list of all files
  * directory/\*.dat
  * bash run_incep_res2_for_dat.sh directory
  * bash run_incep_res2_for_dat.sh \*.dat
* Convert the output to tsv
  * python input output brain_header
  * python ../../data/dat_list/

