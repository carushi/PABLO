import sys
import os
import numpy as np
import subprocess
import nibabel as nib
import argparse

PYRADIOMICS = True
PARAM = 'Params.yaml'
TUMOR = True
# Even if there is a file, recompute the normalized images.
RENEWAL = True

if PYRADIOMICS:
    from radiomics import featureextractor
    # from radiomics import enableCExtensions
    # enableCExtensions(enabled=True)
    import SimpleITK as sitk

if TUMOR:
    R = 1
    ZR = 1
else:
    R = 2  # Resolution
    ZR = 1  # Z-axis resolution
# reproduce the results.
np.random.seed(5)


def get_parser():
    global R, ZR
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', dest='feature', type=str,
                        help='help', default=None)
    parser.add_argument('--indir', dest='dir', type=str)
    parser.add_argument('--outdir', dest='odir', type=str)
    parser.add_argument('--patient', dest='patient', type=str)
    parser.add_argument('--file', dest='file', type=str)
    parser.add_argument('--start', dest='start', type=int, default=0)
    parser.add_argument('--end', dest='end', type=int, default=1000)
    parser.add_argument('--basic', dest='basic', action='store_true')
    parser.add_argument('--atlas', dest='atlas', action='store_true')
    parser.add_argument('--bias', dest='bias', action='store_true')
    parser.add_argument('--renewal', dest='renewal', action='store_true')
    parser.add_argument('--frenewal', dest='frenewal', action='store_true')
    parser.add_argument('--subtract', dest='subtract', action='store_true')
    parser.add_argument('--mask', dest='mask', default=None)
    parser.add_argument('--prefix', dest='prefix', type=str, default=',2w,a')
    parser.add_argument('--low_res', dest='low_res', action='store_true')
    parser.add_argument('--xy_scale', dest='xy_scale', default=R,
                        type=int, help="2: 1/2 resolution (default 2)")
    parser.add_argument('--z_scale', dest='z_scale', default=ZR,
                        type=int, help="2: 1/2 resolution (default 1)")
    return parser


def reduce_resolution(input, output, resolution=3, zresolution=1):
    print(resolution)
    img = nib.load(input)
    data = img.get_data()
    new_img = nib.Nifti1Image(np.array(data[::resolution, ::resolution, ::zresolution]).astype(
        np.int32), img.affine, img.header)
    new_img.header.set_data_dtype('int32')
    nib.save(new_img, output)


class Normalization:
    """Image normalization."""

    def __init__(self, fname, oname=None):
        self.fname = fname
        self.oname = oname
        self.init_params()
        self.set_norm_dict()

    def set_norm_dict(self):
        global PYRADIOMICS
        if PYRADIOMICS:
            self.norm_dict = {'k-method': {'prefix': 'k', 'arg': [0.1, 255]}, 'pyradiomics': {
                'prefix': ['z'], 'arg': [3, 255/3.]}, 'ratio': {'prefix': 'r', 'arg': []}}
        else:
            self.norm_dict = {'k-method': {'prefix': 'k', 'arg': [0.1, 255]}}

    def set_output(self, oname):
        self.oname = oname

    def init_params(self):
        self.t, self.b = [], []
        self.m, self.s = [], []
        self.o = []

    def get_default_output(self, method):
        global TUMOR
        tail = os.path.basename(self.fname) + ('' if '.gz' in self.fname else '.gz')
        file_list = [('' if TUMOR else 'n') + m + tail for m in self.norm_dict[method]['prefix']]
        def add_dirname(v):
            return os.path.join(os.path.dirname(self.fname), v)
        return list(map(add_dirname, file_list)) 

    def clip_each_slice_percent(self, data, outlier=0.5):
        data = data.astype(np.float)
        if outlier == 0.0:
            return data
        self.init_params()
        for z in range(data.shape[2]):
            slice = data[:, :, z]
            b, t = np.percentile(slice.flatten(), (outlier, 100. - outlier))
            slice = np.clip(slice, max(0., b), t)
            data[:, :, z] = slice
            self.b.append(b)
            self.t.append(t)
        return data

    def norm_global_max(self, data, func, scale=100.):
        for z in range(data.shape[2]):
            slice = data[:, :, z]
            if np.std(slice.flatten()) == 0:
                continue
            deno = func(slice.flatten())
            data[:, :, z] = slice / deno * scale
            self.m.append(deno)
        return data

    def norm_each_slice_std(self, data, max_value=1.):
        for z in range(data.shape[2]):
            slice = data[:, :, z]
            if np.std(slice.flatten()) == 0:
                continue
            m, s = np.mean(slice.flatten()), np.std(slice.flatten())
            data[:, :, z] = (slice - m) / s
            self.m.append(m)
            self.s.append(s)
        return data

    def pyradiomics_normalization(self, img, outliers=3, scale=1.):
        ''' outlier=3: 0.26%, 2: 4.6, ...'''
        img = sitk.Normalize(img)
        data = sitk.GetArrayFromImage(img)
        if outliers is not None and outliers > 0.:
            data = np.clip(data, -outliers, outliers)
        data *= scale
        p = data.copy()
        n = data.copy()
        p[np.where(p <= 0.)] = 0
        n[np.where(n >= 0.)] = 0
        n = np.absolute(n)
        z = data.copy()
        z = z-np.min(z.flatten())
        z *= 255./np.max(z.flatten())
        return p, n, z

    def clip_whole_brain(self, data, img_type, outliers=1, scale=255):
        print(min(data.flatten()), max(data.flatten()))
        data[np.isnan(data)] = 0.
        data = data-min(data.flatten())
        value = np.percentile(data.flatten(), 100. - outliers)
        t = np.percentile(value, 100. - outliers)
        data = np.clip(data, 0.0, t)
        print(t, min(data.flatten()), max(data.flatten()))
        assert min(data.flatten()) >= 0. and max(data.flatten()) <= t
        data = data/np.max(data)*scale
        self.m.append(t)
        return data.astype(np.int32)

    def n4_bias_correction(self):
        if self.output is None:
            print('Please specify output file.')
        else:
            subprocess.call('python n4_bias_correction.py ' + input + ' ' +
                            str(3) + ' ' + '[20,20,10,5]' + ' ' + output, shell=True)

    def apply_normalization(self, norm, data, img_type):
        if norm == 'k-method':
            data = self.clip_whole_brain(
                data, img_type, self.norm_dict[norm]['arg'][0])
        # elif norm == 'naldeborgh':
        #     data = self.clip_each_slice_percent(
        #         data, self.norm_dict[norm]['arg'][0])
        #     data = self.norm_each_slice_std(
        #         data, self.norm_dict[norm]['arg'][1])
        elif norm == 'global':
            data = self.clip_each_slice_percent(
                data, self.norm_dict[norm]['arg'][0])
            data = self.norm_global_max(
                data, np.mean, self.norm_dict[norm]['arg'][1])
        # elif norm == '3d-unet':
        #     data = self.clip_each_slice_center(
        #         data, self.norm_dict[norm]['arg'][0])
        #     data = self.norm_each_slice_std(
        #         data, self.norm_dict[norm]['arg'[1]])
        else:
            pass
        return data

    def print_normalization_status(self, norm, prefix):
        print('\t'.join(["#", self.fname, prefix, norm, ' '.join(
            list(map(str, self.norm_dict[norm]['arg'])))]), end='')
        for l, n in zip([self.b, self.t, self.m, self.s], ['b', 't', 'm', 's']):
            if len(l) > 0:
                print('\t', n + ':' + ','.join(list(map(str,
                                                        [np.min(l), np.mean(l), np.max(l)]))), end='')
            else:
                print('\t', end='')
        print('')

    def write_other_img(self, norm, verbose, prefix, xy_scale=1, z_scale=1, img_type=''):
        img = nib.load(self.fname)
        data = img.get_data()
        data = self.apply_normalization(norm, data, img_type)
        if verbose:
            self.print_normalization_status(norm, prefix)
        if self.oname is None:
            return data
        else:
            new_img = nib.Nifti1Image(
                np.array(data[::xy_scale, ::xy_scale, ::z_scale]).astype(np.uint32), img.affine, img.header)
            new_img.header.set_data_dtype('uint32')
            nib.save(new_img, self.oname[0])
            print('Write image:', self.oname)
        return None

    def write_pyradio_img(self, norm, xy_scale=1, z_scale=1):
        iimg = sitk.ReadImage(self.fname)
        pimg, nimg, zimg = self.pyradiomics_normalization(
            iimg, self.norm_dict[norm]['arg'][0], self.norm_dict[norm]['arg'][1])
        if self.oname is None:
            data = sitk.GetArrayFromImage(pimg)
            return data
        for timg, out in zip([pimg, nimg, zimg], self.oname):
            img = nib.load(self.fname)
            new_img = nib.Nifti1Image((np.swapaxes(timg, 0, 2)[
                                      ::xy_scale, ::xy_scale, ::z_scale]).astype(np.int32), img.affine, img.header)
            new_img.header.set_data_dtype('int32')
            nib.save(new_img, out)
            print('Write image:', out)
        return None

    def normalize_image(self, norm, xy_scale, z_scale, verbose=True, prefix="", img_type=''):
        self.init_params()
        img, data = None, None
        print('Apply normalization:', norm, self.oname)
        if norm == 'pyradiomics':
            return self.write_pyradio_img(norm, xy_scale, z_scale)
        else:
            return self.write_other_img(norm, verbose, prefix, xy_scale, z_scale, img_type)


def parse_prefix(fpath):
    fpath = os.path.basename(fpath)
    status = []
    if fpath[0] == 'c':
        status.append('c')
        fpath = fpath[1:]
    elif fpath[0] == 's':
        status.append('s')
        fpath = fpath[1:]
    elif fpath[0] == 'e':
        status.append('e')
        fpath = fpath[1:]
    else:
        status.append('')
        fpath = fpath[1:]
    if fpath[0] == 'a':
        status.append('a')
        fpath = fpath[1:]
    elif fpath[0] == 'w':
        status.append('w')
        fpath = fpath[1:]
    elif fpath[0:2] == '2w':
        status.append('2w')
        fpath = fpath[2:]
    status.extend([fpath.split('.')[0], '.'.join(fpath.split('.')[1:])])
    print(status)
    return status


def get_mask_fname(original, status, mask_default, low_res, patient=''):
    global TUMOR, R, ZR
    if mask_default is None:
        if TUMOR:
            mask = (status[1] + status[2] + '_mask.' + status[3]).strip('.gz')
        else:
            prefix = 'b' + {'a': 'e', '2w': 'e', '': ''}[status[1]]
            mask = (prefix + status[1] + status[2] +
                    '.' + status[3]).strip('.gz')
    else:
        mask = mask_default.replace('[METHOD]', status[2]).replace(
            '[NORM]', status[1]).strip('.gz')
    if '/' in mask:
        if '[PATIENT]' in mask:
            mask = mask.replace('[PATIENT]', patient)
        mdir, mask = os.path.dirname(mask), os.path.basename(mask)
    else:
        mdir = os.path.dirname(original)
    tail = None
    if os.path.exists(os.path.join(mdir, mask + '.gz')):
        tail = '.gz'
    elif os.path.exists(os.path.join(mdir, mask)):
        tail = ''
    else:
        print('no mask image', os.path.join(mdir, mask), '(.gz)')
    if tail is not None:  # found a mask
        if low_res or (mask_default is None and (not TUMOR)):
            reduce_resolution(os.path.join(mdir, mask+tail),
                              os.path.join(mdir, 'n'+mask+tail), R, ZR)
            return 'n' + mask + tail, mdir
        else:
            return mask + tail, mdir
    else:
        return '', ''


def get_corrected_image(original):
    return 'f' + os.path.basename(original.rstrip('.gz')) + '.gz'


def apply_n4_bias_correction(fpath):
    opath = get_corrected_image(fpath)
    subprocess.call('python n4_bias_correction.py \"' + fpath + '\" ' +
                    str(3) + ' ' + '[20,20,10,5]' + ' \"' + opath + '\"', shell=True)
    subprocess.call('mv \"' + os.path.join(os.getcwd(), opath) +
                    '\" \"' + os.path.dirname(fpath) + '\"', shell=True)
    fpath = os.path.join(os.path.dirname(fpath), opath)
    return fpath


def set_command(out, mask, patient, status, plist, feature_out, arg, header):
    cmd = 'python feature_calc.py / ' \
          + out + ' ' + mask + ' --prefix ' + ','.join([patient +
                                                        '_' + '_'.join(status[0:3] + [p]) for p in plist])
    if arg.basic:
        cmd = cmd + ' --basic'
    elif arg.atlas:
        cmd = cmd + ' --atlas'
    if header:
        cmd = cmd + ' --header'
        header = False
    cmd = cmd + ' >> ' + feature_out
    return cmd, header


def norm_slices(patient, feature_out, fpath, arg, brain_types=[], header=True, renewal=False, frenewal=False):
    global TUMOR
    print(brain_types, 'aaa')
    if len(brain_types) == 0:
        brain_types = parse_prefix(fpath)  # parse file name
    assert len(brain_types) == 4
    if brain_types[1] == '' and arg.bias:
        fpath = apply_n4_bias_correction(fpath)
    image_norm = Normalization(fpath, "")
    for m in sorted(image_norm.norm_dict.keys()):
        if m != 'k-method' and m != 'pyradiomics':
            continue
        image_norm.set_output(image_norm.get_default_output(m))
        if len(['' for out in image_norm.oname if not os.path.exists(out)]) > 0 or renewal:
            image_norm.normalize_image(
                m, prefix='\t'.join([patient] + brain_types[0:3]), verbose=False, xy_scale=arg.xy_scale, z_scale=arg.z_scale, img_type=brain_types[1])
        else:
            print('Files exist:', image_norm.oname)
    for m in sorted(image_norm.norm_dict.keys()):
        if m != 'k-method' and m != 'pyradiomics':
            continue
        image_norm.set_output(image_norm.get_default_output(m))
        if len(feature_out) != '':
            for out, p in zip(image_norm.oname, image_norm.norm_dict[m]['prefix']):
                if arg.atlas:
                    tbrain_types = ['', 'r2w']+brain_types[2:]
                    mask, mdir = get_mask_fname(
                        fpath, tbrain_types, arg.mask, arg.low_res, patient)
                else:
                    mask, mdir = get_mask_fname(
                        fpath, brain_types, arg.mask, arg.low_res, patient)
                if os.path.exists(feature_out) and not frenewal:
                    continue
                if arg.atlas:
                    cmd, header = set_command(out, os.path.join(mdir, mask), patient, brain_types, [x for m in sorted(
                        image_norm.norm_dict.keys()) for x in image_norm.norm_dict[m]['prefix']], feature_out, arg, header)
                else:
                    cmd, header = set_command(out, os.path.join(
                        mdir, mask), patient, brain_types, [p], feature_out, arg, header)
                print('Run', cmd)
                subprocess.call(cmd, shell=True)
                if arg.atlas:
                    return


class Subtraction:
    """2 images normalization."""

    def __init__(self, aname, bname, oname):
        self.aname = aname
        self.bname = bname
        self.oname = oname

    def pixel_difference(self):
        aimg = sitk.Normalize(self.aname)
        bimg = sitk.Normalize(self.bname)
        adata = sitk.GetArrayFromImage(aimg)
        bdata = sitk.GetArrayFromImage(bimg)
        new_data = adata-bdata
        return (new_data+255)/2.

    def write_subtracted_images(self, method):
        oimg = self.pixel_difference()
        img = nib.load(self.aname)
        new_img = nib.Nifti1Image(np.swapaxes(
            oimg, 0, 2).astype(np.int32), img.affine, img.header)
        new_img.header.set_data_dtype('int32')
        nib.save(new_img, self.oname)
        print('Write image:', self.oname)


def get_subtraction(method, prefix, arg):
    aimg, bimg = {'1': ('GD', 'T1'), '2': ('T1', 'T2')}[method[1]]
    norm = ('k' if method[0] == 'S' else 'z')
    aimg_path = os.path.exists(os.path.join(prefix, norm+aimg, '.nii'))
    bimg_path = os.path.exists(os.path.join(prefix, norm+bimg, '.nii'))
    oimg_path = os.path.exists(os.path.join(prefix, 'k'+method, '.nii'))
    if os.path.exists(aimg_path) and os.path.exists(bimg_path):
        if not os.path.exists(oimg_path) or arg.renewal:
            sub = Subtraction(aimg_path, bimg_path, oimg_path)
            sub.write_subtracted_images(method)
    elif os.path.exists(aimg_path+'.gz') and os.path.exists(bimg_path+'.gz'):
        if not os.path.exists(oimg_path+'.gz') or arg.renewal:
            sub = Subtraction(aimg_path+'.gz', bimg_path +
                              '.gz', oimg_path+'.gz')
            sub.write_subtracted_images(method)
    else:
        print('To make '+method, 'no original image', aimg, bimg)


def norm_patients(pdir, feature, start, end, arg):
    methods = ['GD', 'T1', 'T2', 'FLAIR', 'GT']
    if arg.subtract:
        methods = ['S1', 'S2', 'Z1', 'Z2']
    for dir in sorted(os.listdir(pdir)):
        if len(dir) != 4:
            continue
        if int(dir) < start:
            continue
        elif int(dir) > end:
            break
        header = True
        for strip in ['']:  # ['', 's', 'c']:
            for pre in arg.prefix.split(','):
                for m in methods:
                    fwrite = arg.frenewal
                    if feature is None:
                        feature_out = ""
                    else:
                        feature_out = os.path.join(arg.odir, '_'.join(
                            [feature, dir, strip, pre, m + ".tsv"]))
                        if arg.frenewal or not os.path.exists(feature_out):
                            fwrite = True
                            with open(feature_out, 'w') as f:
                                f.write('')
                    if m[0] == 'S' or m[0] == 'Z':  # subtracted image
                        get_subtraction(m, os.path.join(
                            pdir, dir, strip + pre), arg)
                    fname = os.path.join(pdir, dir, strip + pre + m + '.nii')
                    if os.path.exists(fname + '.gz'):
                        norm_slices(
                            dir, feature_out, fname + '.gz', arg, [strip, pre, m, 'nii.gz'], header=header, renewal=arg.renewal, frenewal=fwrite)
                    elif os.path.exists(fname):
                        norm_slices(dir,
                                    feature_out, fname, arg, [strip, pre, m, 'nii'], header=header, renewal=arg.renewal, frenewal=fwrite)
                    else:
                        print('No file:', fname)
                        # header = T


if __name__ == '__main__':
    parser = get_parser()
    arg = parser.parse_args()
    if arg.patient is None:
        norm_patients(arg.dir, arg.feature, arg.start, arg.end, arg)
    else:
        norm_slices(arg.petient, os.path.join(arg.dir, arg.file),
                    arg.feature, arg, renewal=arg.renewal, frenewal=arg.frenewal)
