#!/usr/bin/env python
# coding: utf-8

import os
import SimpleITK as sitk
import numpy as np
import sys
import argparse
import nibabel as nib
np.set_printoptions(threshold=np.nan)
from radiomics import featureextractor
from scipy.stats import rankdata, kde
from scipy.ndimage.measurements import center_of_mass
import subprocess

PARAM = 'Params.yaml'


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--header', dest='header', action="store_true")
    parser.add_argument('--basic', dest='basic', action="store_true")
    parser.add_argument('--atlas', dest='atlas', action="store_true")
    return parser

def get_fsl_atlas(lines):
    dict = []
    for line in lines:
        contents = line.strip('\n').split(':')
        if line == '':
            continue
        dict.append((contents[0], contents[1]))
    return dict

def get_ann_ratio(roi):
    atlas_name = ["Harvard-Oxford Subcortical Structural Atlas", "MNI Structural Atlas"]
    hdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/ann/Atlas_annotation/")
    ann_list = []
    prefix = ['har', 'mni']
    files = ["all_harvard_mask.txt", "all_mni_mask.txt"]
    for pre, fname in zip(prefix, files):
        with open(os.path.join(hdir, fname)) as f:
            ann_list.append(dict(get_fsl_atlas(f.readlines())))
    all_dict = {}
    for i, (pre, atlas) in enumerate(zip(prefix, atlas_name)):
        cmd = "atlasquery -a \""+atlas+"\" -m \""+roi+"\""
        print("#", cmd)
        str = subprocess.check_output(cmd, shell=True)
        str = str.decode('utf-8')
        if len(str) == 0:
            all_dict = {**all_dict, **dict([(pre+'_'+key, float(nan)) for key in ann_list])}
        else:
            tdict = dict(get_fsl_atlas(str.split('\n')))
            for key in ann_list[i]:
                all_dict[pre+'_'+key] = (float(tdict[key]) if key in tdict else 0)
    return all_dict


def calc_atlas_feature(pdir, name, mask='', prefix='', header=True):
    path = os.path.join(pdir)
    roi = os.path.join(path, mask)
    if not os.path.exists(roi):
        print('#No data', roi)
        return None
    assert os.path.exists(roi)
    feature = get_ann_ratio(roi)
    print_feature(feature, prefix, header, False)

def get_extractor(param=None):
    global PARAM
    if param is None:
        params = PARAM
    else:
        params = param
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    extractor.settings['geometryTolerance'] = 1e-4
    # extractor.enableCExtensions(True)
    extractor.enableFeatureClassByName('shape')
    return extractor


def otsu_thresholding(image):
    print('# Thresholding...')
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    return otsu_filter.Execute(image)

def calc_feature(pdir, name, mask='', prefix='', header=True):
    extractor = get_extractor()
    path = os.path.join(pdir)
    original = os.path.join(path, name)
    mask = os.path.join(path, mask)
    if not os.path.exists(original):
        print('#No data', original)
        return
    image = sitk.ReadImage(original)
    if os.path.exists(mask):
        label = sitk.ReadImage(mask)
        mdata = sitk.GetArrayFromImage(label).astype(np.int32)
        if len(np.where(mdata >= 1)[0]) < 10:
            mdata = sitk.GetArrayFromImage(label).astype(np.float)
            mdata[np.where(mdata > 0.1)] = 1
            print("#Floating number in mask", len(np.where(mdata >= 1)[0]), mdata.shape)
            mdata = mdata.astype(np.int32)
        nlabel = sitk.GetImageFromArray(mdata)
        nlabel.SetOrigin(label.GetOrigin())
        nlabel.SetSpacing(label.GetSpacing())
        nlabel.SetDirection(label.GetDirection())
        label = nlabel
    elif len(mask) == 0:
        label = otsu_thresholding(image)
        label = sitk.BinaryDilate(label != 0, 3)
    else:
        label = None
    feature = extractor.execute(image, label)
    print_feature(feature, prefix, header)

def calc_basic_feature(pdir, name, mask='', prefix='', header=True):
    path = os.path.join(pdir)
    original = os.path.join(path, name)
    roi = os.path.join(path, mask)
    if not os.path.exists(original):
        print('#No data', original)
        return None
    if mask == '' or not os.path.exists(roi):
        feature = append_image_and_mask(original, None)
    else:
        feature = append_image_and_mask(original, roi)
    print_feature(feature, prefix, header, False)


def mask_image(image, mask):
    sarray = nib.load(image).get_data()
    marray = nib.load(mask).get_data()
    assert sarray.shape == marray.shape
    sarray = sarray.flatten()
    marray = marray.flatten()
    return sarray[np.where(marray > 0)]

def read_nii(ifname):
    image = nib.load(ifname).get_data()
    mass = center_of_mass(image)
    return image.reshape(-1), mass, image.shape

def append_image_and_mask(image, mask, prefix=''):
    all_dict = {}
    img, mass_a, dims = read_nii(image)
    dataset = [img]
    if mask is not None:
        dataset.append(mask_image(image, mask))
        roi, mass_m, _ = read_nii(mask)
        all_dict[prefix + 'size'] = len(np.where(roi > 0.)[0])/max(1, len(np.where(dataset[0] > 0.)[0]))
        if len(np.where(dataset[0] > 0.)[0]) == 0:
            print('# No pixel found', max(dataset[0]))
    for i, array in enumerate(dataset):
        sp = ['all_', 'tumor_'][i]
        all_dict[prefix + sp + 'min'] = np.min(array)
        all_dict[prefix + sp + 'max'] = np.max(array)
        for j, z in enumerate(np.percentile(array, [h for h in range(10, 100, 10)])):
            all_dict[prefix + sp + str(j*10+10) + 'per'] = z
        all_dict[prefix + sp + 'mean'] = np.mean(array)
        if i == 0:
            mass = mass_a
        else:
            mass = mass_m
        for j, x in enumerate(mass):
            all_dict[prefix + sp + 'centro_' + str(j)] = x
        if i == 1:
            for j, x in enumerate(range(3)):
                all_dict[prefix + sp + 'centro_dim_'+str(x)] = dims[j]
    return all_dict

def print_feature(feature, prefix, header=True, general_select=True):
    def no_general_feature(x):
        return not x.startswith("general_info_") and not x.startswith("diagnostics_")
    if general_select:
        feature_name = list(
            sorted(filter(no_general_feature, [x for x in feature])))
    else:
        feature_name = sorted(list(feature.keys()))
    if header:
        print('sample', end='\t')
        print('\t'.join([f for f in feature_name]))
    for p in prefix.split(','):
        print(p, end='\t')
        print('\t'.join([str(feature[f])
                         for f in feature_name]))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.header:
        print('#', ' '.join(sys.argv))
    if len(args.input) == 2:
        if args.basic:
            calc_basic_feature(args.input[0], args.input[1],
                         prefix=args.prefix, header=args.header)
        if args.atlas:
            calc_atlas_feature(args.input[0], args.input[1],
                         prefix=args.prefix, header=args.header)
        else:
            calc_feature(args.input[0], args.input[1],
                         prefix=args.prefix, header=args.header)
    elif len(args.input) > 2:
        if args.basic:
            calc_basic_feature(args.input[0], args.input[1],
                         args.input[2], prefix=args.prefix, header=args.header)
        elif args.atlas:
            calc_atlas_feature(args.input[0], args.input[1],
                         args.input[2], prefix=args.prefix, header=args.header)
        else:
            calc_feature(args.input[0], args.input[1],
                         args.input[2], prefix=args.prefix, header=args.header)
