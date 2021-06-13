# PABLO
PABLO (Publicly Available Brain image anaLysis tOolkit) is a general toolkit for brain MRI analysis.

<img src="https://dl.dropboxusercontent.com/s/8tgfklhvrq9qfjn/logo_pablo.png?dl=0mgv0mmx0p0rctvm/logo_catactor.png?dl=0" width="300">

## Dependency (not fully)
* FSL
* NiBabel
* ANTs
* SPM12
* pyradiomics



## Input

* 3D Nifti file

* DICOM file

  

## Install

Pip install is coming soon to compute image features from Nifti image.

```
pip install PABLO
```



## Contents of PABLO
The full repository, which can be obtained via a command below, contains the scripts used in our study. 

```
git clone https://github.com/carushi/PABLO
```



* Scripts for DICOM conversion into Nifti (SPM)
* Scripts for normalization of image data using SPM and ANTs
* Scripts for skull stripping in each image
* Computation of basic image features
* Computation of image features using pyradiomics
* Construction of classifiers for the data of multiple cohorts

## Image Features 
* Basic features: 
   * feature_calc.py
* Pyradiomic features:
   * feature_calc.py
* Deep learning-based features: Inception-Resnet v2
   * process_nii.py
* Anatomical features:
   * feature_calc.py 


## Reference
Kawaguchi RK., Takahashi M., Miyake M., Kinoshita M., Takahashi S., Ichimura K., Hamamoto R., Narita Y., and Sese J. (to be published)
