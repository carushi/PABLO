import os
import sys
import re
import numpy as np
import nibabel as nib
from PIL import Image
from scipy import signal, interpolate
import scipy.ndimage.interpolation
import cv2
import scipy
import os
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import math

types = ["GT", "GD", "T1", "T2", "FLAIR"]

def contour_snake(image):
    s = np.linspace(0, 2*np.pi, 400)
    x = image.shape[0]/2 + image.shape[0]/2*np.clip(np.cos(s)+0.1, -1, 1)
    y = image.shape[1]/2 + image.shape[1]/2*np.clip(np.sin(s)+0.1, -1, 1)
    init = np.array([x, y]).T
    snake = active_contour(image,
                       init, alpha=0.015, beta=20, gamma=0.001, w_line=0)
    return snake

def get_threshold(np_matrix):
    EPS = 1000
    hist = plt.hist(np_matrix.flatten(), bins=np.arange(0, np.max(np_matrix.flatten()), 5))
    flag = False
    max_value = max(hist[0])
    for i in range(hist[0].shape[0]-1):
        if flag and hist[0][i]+EPS < hist[0][i+1]:
            print(hist[1][i])
            plt.show()
            return hist[1][i]
        if hist[0][i] == max_value:
            flag = True
    return 0

def remove_small_features_strict(np_matrix, imgs):
    for z in range(np_matrix.shape[2]):
        max_value = np.max(np_matrix[:,:,z])
        if max_value == 0.:
            continue
        img = np.clip(np_matrix[:,:,z]/max_value*255., a_min=0, a_max=255)
        img = img.astype(np.uint8)
        ret, imgt = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        imgtn = cv2.morphologyEx(imgt, cv2.MORPH_OPEN, kernel, iterations = 2)
        img_out = np.zeros((imgtn.shape[0], imgtn.shape[1]), dtype=np.uint8)
        snake = contour_snake(gaussian(imgt, 2.5))
        snake = np.array([[p for p in snake]], dtype=np.int32)
        img_out = cv2.fillPoly(img_out, snake, 255)
        imgs[:,:,z] = img_out
    return imgs

def remove_small_features(np_matrix, imgs):
    for z in range(np_matrix.shape[2]):
        max_value = np.max(np_matrix[:,:,z])
        if max_value == 0.:
            continue
        img = np.clip(np_matrix[:,:,z]/max_value*255., a_min=0, a_max=256)
        img = img.astype(np.uint8)
        ret, imgt = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_flood = imgt.copy()
        mask = np.zeros((imgt.shape[0]+2, imgt.shape[1]+2), np.uint8)
        cv2.floodFill(img_flood, mask, (0, 0), 255);
        img_flood_inv = cv2.bitwise_not(img_flood)
        img_out = imgt | img_flood_inv
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_out.astype(np.uint8), connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        min_size = 1000
        img2 = np.zeros((np_matrix.shape[0:2]))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                print(z, i, sizes[i])
                img2[output == i + 1] = 1
        imgs[:,:,z] = img2
    return imgs

def write_binarized_nifti_spm(ifname, ofname, masked='', reduced=True, threshold=0.5):
    template = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../ref/spm12/tpm/TPM.nii')
    img = nib.load(ifname)
    header = img.header
    affine = img.affine
    data = img.get_data()
    imgs = np.zeros(
        shape=(data.shape[0], data.shape[1], data.shape[2]), dtype=np.int16)
    x, y, z = data.shape[0], data.shape[1], data.shape[2]
    tpm = nib.load(template).get_data()[:,:,:,0:3]
    tpm_interpolated = scipy.ndimage.interpolation.zoom(tpm, (x/tpm.shape[0], y/tpm.shape[1], z/tpm.shape[2], 1), order=1)
    imgs[np.where(tpm_interpolated.sum(axis=3) > threshold)] = 1
    new_img = nib.Nifti1Image(np.array(imgs), affine, header)
    print('writing...', os.getcwd(), ofname)
    nib.save(new_img, ofname)
    if len(masked) > 0:
        imgs_masked = data
        imgs_masked[np.where(imgs == 0)] = 0
        print('writing...', os.getcwd(), masked)
        new_img = nib.Nifti1Image(np.array(imgs_masked), affine, header)
        nib.save(new_img, masked)



def write_binarized_nifti(ifname, ofname, reduced=True, min_t=0.):
    img = nib.load(ifname)
    header = img.header
    affine = img.affine
    data = img.get_data()
    imgs = np.zeros(
        shape=(data.shape[0], data.shape[1], data.shape[2]), dtype=np.int16)
    if reduced:
        imgs = remove_small_features_strict(data, imgs)
    else:
        imgs[np.where(np.isnan(data))] = 0
        imgs[np.where(data > min_t)] = 1
        imgs[np.where(data <= min_t)] = 0
    new_img = nib.Nifti1Image(np.array(imgs), affine, header)
    print('writing...', os.getcwd(), ofname)
    nib.save(new_img, ofname)

def mask_binary(niidir):
    files = os.listdir(niidir)
    for type in types:
        for prefix in ['w', '2w', 'a']:
            for suffix in ['.gz', '']:
                file = prefix + type + '_mask.nii' + suffix
                if file in files:
                    print("reading...", niidir, file)
                    write_binarized_nifti(os.path.join(niidir, file), os.path.join(
                        niidir, 'b' + prefix + type + '_mask.nii' + suffix), False, (0.1 if 'w' in prefix else 0.))
                    break
                else:
                    print('no', file)

def main(niidir, stripped_flag='tcia'):
    files = os.listdir(niidir)
    print('search directory', niidir)
    for type in types:
        for prefix in ['', 'w', '2w', 'a']:
            if prefix not in ['w', '2w']:
                for suffix in ['.gz', '']:
                    if stripped_flag in os.getcwd():
                        file = ('' if prefix == '' else 'a' if prefix == 'a' else '') + type + '.nii' + suffix
                    else:
                        file = ('s' if prefix == '' else 'ea' if prefix == 'a' else '') + type + '.nii' + suffix
                    if file in files:
                        print("reading...", niidir, file)
                        write_binarized_nifti(os.path.join(niidir, file), os.path.join(
                            niidir, 'b' + prefix + type + '.nii' + suffix), False)
                        break
                    else:
                        print('no', file)
            else:
                for suffix in ['.gz', '']:
                    file = prefix + type + '.nii' + suffix
                    if file in files:
                        print("reading...", niidir, file)
                        write_binarized_nifti_spm(os.path.join(niidir, file), os.path.join(
                            niidir, 'b' + prefix + type + '.nii' + suffix), os.path.join(niidir, 's' + file))
                        break
                    else:
                        print('no', file)

if __name__ == "__main__":
    if sys.argv[1] == '--mask':
        mask_binary(sys.argv[2].rstrip('\"').lstrip('\"'))
    else:
        if len(sys.argv) > 2:
            stripped_flag = sys.argv[2]
        else:
            stripped_flag = 'tcia'
        main(sys.argv[1].rstrip('\"').lstrip('\"'), stripped_flag)
