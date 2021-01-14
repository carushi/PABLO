import SimpleITK as sitk
import nibabel as nib
import subprocess
import os
import shutil
import numpy as np
import sys

def make_dir(tdir):
    if not os.path.exists(tdir):
        os.makedirs(tdir)

def get_file_prefix(sdir):
    osdir = None
    if 'flair' in sdir:
        osdir = "FLAIR"
    elif 'Gd' in sdir:
        osdir = "GD"
    elif 't1' in sdir:
        osdir = "T1"
    elif 't2' in sdir:
        osdir = "T2"
    elif 'Boost' in sdir:
        if 'Manually' in sdir:
            osdir = 'MROI'
        else:
            osdir = 'AROI'
    return osdir

def set_mask(input, output, func):
    img = nib.load(input)
    header = img.header
    affine = img.affine
    data = img.get_data()
    pos = np.where(func(data))
    data[:,:,:] = 0
    data[pos] = 1
    print(np.max(data.flatten()), np.min(data.flatten()))
    new_img = nib.Nifti1Image(np.array(data), affine, header)
    nib.save(new_img, output)
    subprocess.call('gzip \"' + output + "\"", shell=True)

def make_roi_nii(idir, out, prefix):
    header = ('' if prefix == 'MROI' else 'g')
    # data = ['others','necrosis', 'edema', 'net', 'et'] #0, 1, 2, 3, 4
    set_mask(idir, os.path.join(out, 'GT_'+header+'mask.nii'), np.vectorize(lambda x: x > 0))
    set_mask(idir, os.path.join(out, 'GD_'+header+'mask.nii'), np.vectorize(lambda x: x == 1 or x == 4))

def copy_image_data(source, out):
    shutil.copy(source, out)
    print('Copy data from', source, 'to', out)

def convert_directory(idir, odir):
    id = 1
    make_dir(odir)
    print(idir)
    for root, subdirs, files in os.walk(idir):
        files =[f for f in files if 'nii.gz' in f and f[0] != '.']
        new_id = str(id).rjust(4, '0')
        if len(files) < 5:
            continue
        print(id, new_id, root, files)
        # if id < 62:
        #     id = id+1
        #     continue
        make_dir(os.path.join(odir, new_id))
        print('id', idir.split('/')[-1], root, odir.split('/')[-1], new_id)
        for source in files:
            prefix = get_file_prefix(source)
            if prefix is not None:
                if 'ROI' in prefix:
                    make_roi_nii(os.path.join(idir, root, source), os.path.join(odir, new_id), prefix)
                else:
                    copy_image_data(os.path.join(idir, root, source), os.path.join(odir, new_id, prefix+'.nii.gz'))
                    if prefix == 'GD':
                        copy_image_data(os.path.join(idir, root, source), os.path.join(odir, new_id, 'GT'+'.nii.gz'))

        id += 1

def read_directory():
    with open('directory_list.txt') as f:
        flag = False
        for line in f.readlines():
            if flag:
                return line.rstrip('\n')
            if line.rstrip('\n') == '%tciadir':
                flag = True
    return None

if __name__ == '__main__':
    dir = read_directory()
    if len(sys.argv) > 2:
        input, output = sys.argv[1], sys.argv[2]
        convert_directory(os.path.join(dir, input),os.path.join(dir, output))
    # convert_directory(os.path.join(dir, "LGG"),odir+"_glioma")
    # convert_directory(os.path.join(dir, "GBM"),odir+"_gbm")
