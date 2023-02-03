# PABLO
PABLO (Publicly Available Brain image anaLysis tOolkit) is a general toolkit for brain MRI analysis.

<img src="https://dl.dropboxusercontent.com/s/8tgfklhvrq9qfjn/logo_pablo.png?dl=0mgv0mmx0p0rctvm/logo_catactor.png?dl=0" width="300">



PABLO is a versatile platform for analyzing combinations of methods consisting of three steps widely usde in MRI analysis; 1) spatial standardization, 2) dimension reduction and 3) classification. Because of the complex input and output, the pipeline is not fully automated. This repository contains the scripts for all steps used in our study. In mini PABLO, which can be downloaded through pip, only the scripts to compute image features are available.



## Dependency (not fully)

* FSL
* NiBabel
* ANTs
* SPM12
* pyradiomics



## Install

Pip install is coming soon to compute image features from Nifti image.

```
pip install PABLO-MRI

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

## Image Features available in mini PABLO
* Basic features: 
   * Source: feature_calc.py
   * Option: basic
* Pyradiomic features
   * Source: feature_calc.py
   * Option: py
   * [Pyradiomics manual](https://pyradiomics.readthedocs.io/en/latest/)
* Deep learning-based features: Inception-Resnet v2
   * Source: process_nii.py
   * Additional requirement: keras model
   * Option: deep
* Anatomical features
   * feature_calc.py 
   * Option: ann

## Input

* 3D Nifti file
* DICOM file




## Reference
Kawaguchi RK, Takahashi M, Miyake M, Kinoshita M, Takahashi S, Ichimura K, Hamamoto R, Narita Y, Sese J. Assessing Versatile Machine Learning Models for Glioma Radiogenomic Studies across Hospitals. <i>Cancers (Basel)</i>. 2021 Jul 19;13(14):3611. doi: 10.3390/cancers13143611. PMID: 34298824; PMCID: PMC8306149.
