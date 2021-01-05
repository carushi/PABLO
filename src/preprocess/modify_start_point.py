# Convert q and s form (used for TCIA)

import os
import nibabel as nib
import sys
import numpy as np
import re

def convert_to_new_origin(fname):
    ofname = re.sub('.nii|.nii.gz', '', fname)+'_changed.nii'
    if os.path.exists(ofname):
        os.remove(ofname)
    img = nib.load(fname)
    header = img.header
    affine = img.affine
    data = img.get_data()
    affine[0,0] = -1
    affine[1,1] = -1
    affine[2,2] = 1
    affine[:,3] = [(data.shape[0])/2., (data.shape[1])/2., -(data.shape[2])/2., 1]
    new_data = nib.Nifti1Image(np.array(data), affine, header)
    new_data.header['qform_code'] = 0
    new_data.header['sform_code'] = 1
    new_data.header['srow_x'] = np.array([-1, 0, 0, (data.shape[0])/2.])
    new_data.header['srow_y'] = np.array([0, -1, 0, (data.shape[1])/2.])
    new_data.header['srow_z'] = np.array([0, 0, 1, -(data.shape[2])/2.])
    new_data.header['quatern_b'] = 0.
    new_data.header['quatern_c'] = 0.
    new_data.header['quatern_d'] = 0.
    new_data.header['qoffset_x'] = 0.
    new_data.header['qoffset_y'] = 0.
    new_data.header['qoffset_z'] = 0.
    new_data.header['pixdim'] = [1,1,1,1,1,1,1,1]
    print('writing...', ofname)
    nib.save(new_data, ofname)
    out = nib.load(ofname)
    print(fname, ofname)
    #new_data.header['sform_code'] = 1
    print(new_data.header['sform_code'])
    # for key in new_data.header:
    #     print(key, new_data.header[key])
    assert out.header['sform_code'] == 1.0
    assert out.header['quatern_d'] == 0.

argvs = sys.argv
for dir in os.listdir(argvs[1]):
    full_path = os.path.join(argvs[1], dir)
    print(full_path)
    if not os.path.isdir(full_path): continue
    print(dir)
    for file in [f for f in os.listdir(full_path)]:
        print(file)
        if 'nii.gz' in file:
            os.system('gunzip -f '+os.path.join(full_path, file))
        if 'nii' in file and 'changed' not in file:
            convert_to_new_origin(os.path.join(full_path, file.replace('.gz', '')))
