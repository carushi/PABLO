#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import zipfile
import pandas as pd
import seaborn as sns
import sys
import re
import math
from copy import deepcopy


from collections import OrderedDict
from classification_base import *

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
from scipy.stats import wilcoxon, ranksums, stats
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model

from whitening import *
import xgboost as xgb
import nibabel as nib

VALIDATION = ['oppose', 'validation', 'both', 'test']
FVALIDATION = ['both', 'test', 'validation', 'oppose']

def plot_corr_features(feature_data, feature_name, sample_name, fname):
    print(feature_data.shape, len(feature_name), len(sample_name))
    mat = feature_data
    d = pd.DataFrame(data=mat, columns=feature_name, index=sample_name)
    print(max(d.max()), min(d.min()))
    if feature_data.shape[1] < 5000:
        fig, ax = plt.subplots()
        ax.grid(False)
        plt.imshow(mat)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('original_heat_' + fname + '_sample.pdf')
        plt.close(fig)
        plt.clf()
        samples = StandardScaler().fit_transform(remove_nan(mat).real)
        fig, ax = plt.subplots()
        ax.grid(False)
        plt.imshow(samples)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('scale_heat_' + fname + '_sample.pdf')
        plt.close(fig)
        plt.clf()

    corr = d.corr()
    fig, ax = plt.subplots()
    ax.grid(False)
    plt.imshow(corr)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('corr_' + fname + '_sample.pdf')
    plt.close(fig)
    plt.clf()
    sample_order = plot_dendrogram(d, fname, sample_name)
    return sample_order


def plot_dendrogram(d, gname, index):
    from scipy.cluster.hierarchy import dendrogram, linkage
    import seaborn as sns
    from scipy.spatial.distance import pdist
    print(d.head())
    m = d.values
    Z = linkage(pdist(m), method='complete')
    Z[np.isinf(Z)] = 1e+6
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels=index
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.savefig(('hclust_' + gname + '.pdf').replace('/', ''))
    plt.close()
    plt.clf()
    if gname != '-1':
        pp = sns.clustermap(d, col_cluster=False,
                            metric='chebyshev', z_score=1)
        _ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)
        plt.savefig(('hclust_heat_' +
                     gname + '.pdf').replace('/', ''))
        plt.close()
        plt.clf()
    return dendrogram(Z)['ivl']

def plot_ann_clustering(sample_order, full_annotations, annotation_names, grade_list, header):
    global FEATURE
    selected = ['GBM', 'CpG', 'MGMT', 'IDH1_2', 'TERT', '1p19q']
    colors = ['Greys', 'Reds', 'Reds', 'Blues', 'Greens', 'Oranges']
    mat = None
    count = -1
    for i, key in enumerate(FEATURE):
        if key not in selected: continue
        count += 1
        if count >= len(colors):
            break
        fig, ax = plt.subplots()
        ax.grid(False)
        index = np.array([grade_list.index(x) for x in sample_order])
        if mat is None:
            mat = np.zeros((len(index), len(colors), 4), dtype=np.float)
        cmap = plt.get_cmap(colors[count])
        for j, x in enumerate([cmap(min(x, 0.5)) for x in full_annotations[i,index].reshape((1, len(index)))[0,:].tolist()]):
            mat[j,count,:] = x
        if i == 0:
            ax.imshow(full_annotations[i,index].reshape((1, len(index))), interpolation='nearest', cmap=plt.get_cmap(colors[count]))
            plt.savefig('ann_heatmap_' + header + '_' + key + '.pdf')
            plt.clf()
    if count == -1:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.grid(False)
    ax.imshow(mat, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ann_heatmap_' + header + '_' + 'all' + '.pdf')
    plt.clf()

def plot_roc(feature, annotation, header):
    pos = 1
    index = [i for i in range(len(annotation)) if annotation[i] == annotation[i]]
    tann, tfea = [annotation[i] for i in index], feature[np.array(index)]
    fpr, tpr, thresholds = roc_curve(tann, remove_nan(tfea), pos_label=pos)
    auroc = auc(fpr, tpr)
    if auroc < 0.5:
        pos = 0
        fpr, tpr, thresholds = roc_curve(tann, remove_nan(tfea), pos_label=pos)
        auroc = 1. - auroc
    plt.plot(fpr, tpr)
    plt.tight_layout()
    plt.title("positive="+str(pos)+", auroc="+str(auroc))
    plt.savefig('auc_fet_' + header + '.pdf')
    plt.close()
    plt.clf()
    with open('roc_objects_'+header+'.pyn', 'wb') as f:
        pickle.dump({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds}, f)


def plot_boxplot(feature, annotation, header):
    global FEATURE
    d = pd.DataFrame(data={'feature': feature, 'ann':annotation})
    ax = sns.boxplot(x="ann", y="feature", data=d)
    plt.savefig('boxplot_fet_' + header + '.pdf')
    plt.close()
    plt.clf()
    d = pd.DataFrame(data={'feature': feature, 'ann':annotation})
    ax = sns.boxplot(x="ann", y="feature", data=d, showfliers=False)
    plt.savefig('boxplot_out_fet_' + header + '.pdf')
    plt.close()
    plt.clf()


class CentroidConverter:
    def __init__(self, ants_flag):
        self.ants_flag = ants_flag
        self.map_data = self.read_prob_map()
        self.map_func = None
        self.dims = []

    def read_prob_map(self):
        global ANTSTEMP, SPMTEMP
        data = None
        if self.ants_flag:
            template = ATNSTEMP
            for i in range(1, 7):
                fname = template.replace('[NUM]', str(i))
                img = nib.load(fname).get_data()
                if data is None:
                    data = img[:,:,:,None]
                else:
                    data = np.append(data, img[:,:,:,None], axis=-1)
        else:
            template = SPMTEMP
            data = nib.load(template).get_data()
        return data

    def scale_template(self, dims):
        func = []
        for j in range(self.map_data.shape[3]):
            x, y, z = self.map_data.shape[0], self.map_data.shape[1], self.map_data.shape[2]
            func.append(RegularGridInterpolator((np.linspace(0, dims[0], x), np.linspace(0, dims[1], y), np.linspace(0, dims[2], z)), self.map_data[:,:,:,j]))
        self.map_func = func
        self.dims = dims


    def get_3d_prof(self, pos_list, dims):
        pos_list = [pos_list[0], [0, 0, 0]]
        if self.map_func is None:
            self.scale_template(dims[0])
        feature_vec, name_vec = [], []
        for j in range(self.map_data.shape[3]):
            feature_vec.append(self.map_func[j]([[x,y,z] for i, (x, y, z) in enumerate(pos_list)]))
        feature_vec = [x for x in zip(*feature_vec)]
        name_vec = [['centroid_'+str(j) for j in range(self.map_data.shape[3])]]*len(pos_list)
        return feature_vec, name_vec


class FeatureExtrcation:
    def __init__(self, arg):
        self.arg = arg

    def read_resnet_data(self, fname, patient, method, grade, validation):
        dir = (self.arg.dir[1] if validation else self.arg.dir[0])
        fname = replace_patient_method(fname, patient, method, grade)
        names, features = [], []
        if not os.path.exists(os.path.join(dir, fname)):
            print('no file', os.path.join(dir, fname))
            return [], []
        print(dir, fname)
        with open(os.path.join(dir, fname)) as f:
            lines = [line.strip('\n').split('\t') for line in f.readlines()]
            assert len(lines) > 0
            names = lines[0][1:]
            features = lines[1][1:] # Remove the first element (sample name)
        return features, names

    def read_feature_data_standard(self, fname, patient, method, grade, validation, converter):
        if validation:
            dir, norm = self.arg.dir[1], self.arg.normalize[1]
        else:
            dir, norm = self.arg.dir[0], self.arg.normalize[0]
        fname = replace_patient_method(fname, patient, method, grade)
        sample = replace_patient_method(norm, patient, method, grade, fname.split('_')[3])
        name = []
        if not os.path.exists(os.path.join(dir, fname)):
            print('no file', os.path.join(dir, fname))
            return [], []
        print(dir, fname, sample)
        with open(os.path.join(dir, fname)) as f:
            for i, line in enumerate([line for line in f.readlines() if len(line) > 0 and line[0] != "("]):
                contents = line.rstrip('\n').split('\t')
                if i == 0:
                    name = contents.copy()
                else:
                    if contents[0] == sample:
                        while '' in contents:
                            contents.remove('')
                        features = [float(contents[i]) for i in range(len(name)) if "tumor_centro_" in name[i]]
                        if len(features) == 6:
                            feature_vec, name_vec = converter.get_3d_prof([features[0:3]], [features[3:]])
                            return feature_vec[0], name_vec[0]
        return [], []

    def read_feature_data_pyradiomics(self, fname, patient, method, grade, validation):
        if validation:
            dir, norm = self.arg.dir[1], self.arg.normalize[1]
        else:
            dir, norm = self.arg.dir[0], self.arg.normalize[0]
        fname = replace_patient_method(fname, patient, method, grade)
        sample = replace_patient_method(norm, patient, method, grade, fname.split('_')[-2])
        name = []
        dir_fname = os.path.join(dir, fname)
        if not os.path.exists(dir_fname):
            print('no file', dir_fname)
            return [], []
        print(dir_fname, sample)
        with open(dir_fname) as f:
            for i, line in enumerate([line for line in f.readlines() if line[0] not in ['(', '#']]):
                contents = line.rstrip('\n').split('\t')
                if i == 0:
                    name = contents.copy()
                else:
                    if contents[0] == sample:
                        # print(len(contents), len(name))
                        while '' in contents:
                            contents.remove('')
                        index = [i for i in range(len(name)) if i > 0 and "tumor_centro_" not in name[i]]
                        if max(index) >= len(contents):
                            print('Caution: truncated features', fname)
                        return [float(contents[i]) if i < len(contents) else float('nan') for i in index], [name[i] for i in index]
        return [], []

    def get_patient_list(self, fname):
        patient_list = {}
        flag = True
        with open(fname) as f:
            for line in f.readlines():
                contents = line.rstrip('\n').split('\t')
                if flag:
                    print(contents, '...')
                    flag = False
                if contents[0] not in patient_list:
                    patient_list[contents[0]] = []
                patient_list[contents[0]].append(contents[1])
        for key in patient_list:
            patient_list[key] = sorted(patient_list[key])
        return patient_list

    def read_reg_files(self, fname):
        assert os.path.exists(fname)
        feature_files, tags = [], []
        with open(fname) as f:
            for line in f.readlines()[1:]:
                if line[0] == "#":
                    continue
                contents = line.rstrip('\n').split(' ')
                if len(contents) != 2:
                    print('# Warning: truncation', contents)
                    continue
                feature_files.append(contents[0])
                tags.append(contents[1])
        return feature_files, tags

    def read_feature_files(self, validation=False):
        if not validation and self.arg.reg_file[0] is not None:
            feature_files, tags = self.read_reg_files(self.arg.reg_file[0])
        elif validation and self.arg.reg_file[1] is not None:
            feature_files, tags = self.read_reg_files(self.arg.reg_file[1])
        else:
            feature_files = self.arg.feature.split(',')+[self.resnet_regexp[(1 if validation else 0)]]
            tags = ['pyradiomics_'+str(i) for i in range(len(feature_files)-1)]+['resnet']
            if self.arg.tumor_map is not None and len(self.arg.tumor_map) > 0:
                feature_files += [self.arg.tumor_map]
                tags.append('centroid')
        return feature_files, tags

    def get_feature_data(self, patient_list, validation=False, feature_names=[], feature_lib=FeatureLibrary()): #feature_files=None,
        global METHODS, SAMPLE
        print(len(feature_names))
        if len(feature_names) > 0:
            validation, index = True, 1
        else:
            validation, index = False, 0
        count = 0
        feature_data, grade_list, feature_status = [], [], []
        feature_files, tags = self.read_feature_files(validation)
        if 'centroid' in tags:
            converter = CentroidConverter('_a_' in feature_files[tags.index('centroid')])
        N = sum([len(patient_list[key]) for key in patient_list])
        print(patient_list.keys())
        for grade in sorted(list(patient_list.keys())):
            for i, patient in enumerate(patient_list[grade]):
                if i % 10 == 0: print("# Reading", grade, patient)
                array, name, status = [], [], []
                if grade in METHODS:
                    methods = METHODS[grade]
                else:
                    methods = METHODS[('gbm' if 'gbm' in grade else 'glioma')]
                tissue_flag = False
                for method in methods:
                    for j, (prefix, tag) in enumerate(zip(feature_files, tags)):
                        if tag == 'tissue':
                            if tissue_flag: continue
                            feature, label = self.read_feature_data_pyradiomics(prefix, patient, method, grade, validation)
                            fcount = feature_lib.insert([tag])
                            tissue_flag = True
                        else:
                            if tag == 'resnet':
                                feature, label = self.read_resnet_data(prefix, patient, method, grade, validation)
                            elif tag == 'centroid':
                                feature, label = self.read_feature_data_standard(prefix, patient, method, grade, validation, converter)
                            else:
                                feature, label = self.read_feature_data_pyradiomics(prefix, patient, method, grade, validation)
                            fcount = feature_lib.insert([(method if method != 'GT' else 'GD'), tag])
                        if len(feature) == 0 and (tag not in ['resnet', 'centroid', '']):
                            print('no read', prefix, tag, grade, method, patient, self.arg.normalize)
                            label = []
                        assert len(feature) == len(label)
                        if tag == 'tissue':
                            name.extend([l for l in label])
                        else:
                            name.extend([l+'_'+(method if method != 'GT' else 'GD') for l in label])
                        array.extend(feature)
                        status.extend([fcount]*(len(feature)))
                if patient == '0001' and grade == 'gbm': # Set matrix according to the data of 0001 gbm patient
                    M = len(name)
                    feature_names, feature_status, feature_data = name[:], status[:], np.repeat(np.nan, N*M).reshape((N, M))
                    print('feature length', M, fcount)
                    for i, key in enumerate(feature_lib.dicts):
                        print(key, len([x for x in feature_status if x == i]))
                    if len([x == x for x in feature_status]) == 0:
                        sys.exit('No feature obtained... check input files and regex.')
                elif patient == '0001' and 'gbm' in grade: # Validation set
                    assert validation
                    M = len(feature_names)
                    print('feature length', M, fcount)
                    feature_data = np.repeat(np.nan, N*M).reshape((N, M))
                feature_data[count,:] = np.array([ float(array[name.index(f)]) if f in name else float('nan') for f in feature_names])
                grade_list.append(grade+'_'+patient)
                count += 1
        return feature_data, grade_list, feature_names, feature_status, feature_lib


class Annotation:
    def __init__(self, names, vnames=None):
        global FEATURE
        self.keys = FEATURE
        self.names = names #gbm_0001, glioma_0001, ...
        self.vnames = vnames
        self.annotation = None

    def read_ann_data(self, afname):
        ann_name, data, annotated_patient = [], [], []
        with open(afname) as f:
            for line in f.readlines():
                contents = line.rstrip('\n').split('\t')
                if len(ann_name) == 0:
                    ann_name = contents
                    for key in contents:
                        data.append([])
                else:
                    if len(contents) < 3 or len(contents[2]) == 0:
                        continue
                    c1, c2 = ann_name.index('Dir'), ann_name.index('Patients')
                    if contents[c2] == 'nan':
                        continue
                    annotated_patient.append(contents[c1]+'_'+'{0:0>4}'.format(int(contents[c2].lstrip('NCC'))))
                    for i, c in enumerate(contents):
                        data[i].append(c)
        return ann_name, data, annotated_patient

    def get_annotation_feature(self, afname, ann_feature):
        ann_name, data, annotated_patient = self.read_ann_data(afname)
        keys = ann_feature.split(',')
        mat = np.repeat(np.nan, len(keys)*len(self.names)).reshape((len(self.names), len(keys)))
        vmat = None
        for i, key in enumerate(keys):
            mat[:, i] = self.convert_ann_to_feature(key, data[ann_name.index(key)], annotated_patient, self.names)
        if self.vnames is not None and len(self.vnames) > 0:
            vmat = np.repeat(np.nan, len(keys)*len(self.vnames)).reshape((len(self.vnames), len(keys)))
            for i, key in enumerate(keys):
                vmat[:, i] = self.convert_ann_to_feature(key, data[ann_name.index(key)], annotated_patient, self.vnames)
        return keys, mat, vmat

    def convert_ann_to_feature(self, key, vector, annotated_patient, names):
        float_flag, dict = False, {}
        keys = key.split('_')
        if len(keys) > 1 and keys[1] == 'n':
            float_flag = True
            # numerical value
        else:
            for i, x in enumerate(sorted(list(set(vector)))):
                if x == '' or x == 'NA':
                    continue
                dict[x] = i
            print(key, 'convert', dict)
            # associate with factors
        data = []
        for patient in names:
            index = annotated_patient.index(patient)
            if vector[index] == 'NA' or len(vector[index]) == 0:
                data.append(float('nan'))
            elif float_flag:
                data.append(float(vector[index]))
            else:
                data.append(dict[vector[index]])
        return data

    def get_positive_negative_matrix(self, afname, names=None):
        if names is None:
            names = self.names
        return self.get_positive_negative_matrix(afname, names)


    def get_positive_negative_matrix(self, afname, names):
        ann_name, data, annotated_patient = self.read_ann_data(afname)
        mat = np.repeat(np.nan, len(self.keys)*len(names)).reshape((len(self.keys), len(names)))
        dict = {'GBM':['Diag'], 'CpG':['16CpG'], 'MGMT':['16CpG'], 'IDH1_2':['IDH1/2'], 'Group A':['Group'], 'Group B':['Group'], 'Group C':['Group'], 'Group D':['Group'], \
                'MGMT GBM':['Diag', '16CpG'], 'MGMT LGG':['Diag', '16CpG'], 'IDH GBM':['Diag', 'IDH1/2'], 'TERT GBM':['Diag', 'TERT'], 'TERT LGG':['Diag', 'TERT'], 'IDH LGG':['Diag', 'IDH1/2'],}
        for i, key in enumerate(self.keys):
            if key == 'GBM' and dict[key] not in ann_name:
                mat[i,:] = [1 if 'gbm' in names[i] else 0 if 'glioma' in names[i] else float('nan') for i in range(len(names))]
            else:
                column = ([key] if key not in dict else dict[key])
                break_flag = False
                for c in column:
                    if c not in ann_name:
                        break_flag = True
                if not break_flag:
                    mat[i,:] = [self.set_ann_threshold(key, [data[ann_name.index(c)] for c in column], annotated_patient, patient) for patient in names]
        print(names[0])
        if 'tcia' in names[0]:
            mat = self.fill_group(mat)
        return mat, self.keys

    def fill_group(self, mat):
        global FEATURE
        if 'IDH1_2' not in FEATURE or 'TERT' not in FEATURE:
            return mat
        idh, tert = FEATURE.index('IDH1_2'), FEATURE.index('TERT')
        for i, a in enumerate(FEATURE):
            if 'Group' not in a:    continue
            if a == 'Group A':
                mat[i,:] = [1 if mat[idh,j] == 1 and mat[tert,j] == 1 else 0 for j in range(mat.shape[1])]
            elif a == 'Group B':
                mat[i,:] = [1 if mat[idh,j] == 1 and mat[tert,j] == 0 else 0 for j in range(mat.shape[1])]
            elif a == 'Group C':
                mat[i,:] = [1 if mat[idh,j] == 0 and mat[tert,j] == 0 else 0 for j in range(mat.shape[1])]
            else: #a == 'Group D'
                mat[i,:] = [1 if mat[idh,j] == 0 and mat[tert,j] == 1 else 0 for j in range(mat.shape[1])]
        return mat

    def set_ann_threshold(self, key, vector, annotated_patient, patient):
        index = annotated_patient.index(patient)
        for c in range(len(vector)):
            if vector[c][index] == 'NA' or len(vector[c][index]) == 0:
                return float('nan')
        value = vector[-1][index]
        if key == 'CpG' and float(value) > 0.0: return 1
        if key == '1p19q' and (value != 'no loss' and value != 'non-codel'):
            return 1
        if 'Group' in key:
            if value == key: return 1
        if 'GBM' in key:
            if vector[0][index] != 'GBM':
                return float('nan')
        if 'LGG' in key:
            if vector[0][index] == 'GBM':
                return float('nan')
        if 'MGMT' in key:
            if float(value) >= 16.0:
                return 1
        if 'IDH1_2' in key or 'IDH' in key:
            if value.lower() != 'wt': return 1
        if 'TERT' in key:
            if value.lower() != 'wt': return 1
        return 0

class SupervisedClassification:
    def __init__(self, X, Y, auc, CV=10):
        global SEED
        self.X, self.Y = self.filt_data(X, Y)
        self.CV = min(CV, len([y for y in self.Y if y > 0]))
        self.auc = auc
        if self.auc:
            self.metrics = ['roc_auc', 'precision', 'recall', 'accuracy']
        else:
            self.metrics = ['accuracy', 'precision', 'recall']
        self.valid = False
        self.vx, self.vy = None, None
        self.sample = StratifiedKFold(n_splits=self.CV, shuffle=True, random_state=SEED)
        self.test_array = None
        print(np.max(self.X), np.min(self.X))

    def filt_data(self, X, Y):
        Y = [1 if y == 'high' or y == 1 else 0 if y == 'low' or y == 0 else np.nan for y in Y]
        filt = np.array([i for i in range(len(Y)) if Y[i] == Y[i]])
        if filt.shape[0] == 0: # No valid data
            return None, None
        return X[filt, :], np.array([y for y in Y if y == y])

    def set_valid_dataset(self, vx, vy):
        self.vx, self.vy = self.filt_data(remove_nan(vx), vy)
        if self.vx is not None:
            self.valid = True

    def divide_test_set(self, key):
        global SEED
        print(key)
        fname = "ann_ncc_test_validation_"+key+".txt"
        # print(fname)
        full_path = os.path.join("/Users/cawa/Research/GBM/script/ncc_pipeline/test/ann/", fname)
        if not os.path.exists(full_path):
            skf = StratifiedKFold(n_splits=2, random_state=SEED, shuffle=True)
            for train_index, test_index in skf.split(self.vx, self.vy):
                # print(len(self.vy), len(train_index), len(test_index))
                assert abs(len(train_index)-len(test_index)) <= 2
                with open(full_path, 'w') as f:
                    for i in train_index:
                        f.write(str(i)+'\n')
        with open(full_path) as f:
            self.test_array = np.array([int(line.rstrip('\n')) for line in f.readlines() if line != ''])

    def print_sparse_features(self, method, header, coef):
        # print(coef[0])
        # print('selected_features', method, header)
        # print(header.lower)
        # print(len([x for x in coef[0] if abs(x) > 0.01]))
        # print(len([x for x in coef[0] if abs(x) > 0.0 ]))
        print('selected_features', method, header.lower(), len([x for x in coef[0] if abs(x) > 0.01]), len([x for x in coef[0] if abs(x) > 0.0 ]))

    def test_validation_set(self, method, header, clf, plot_prefix='', plot_roc=True, pred_y=None, probas=None):
        global SEED
        if 'lda' in method or 'lasso' in method:
            self.print_sparse_features(method, header, clf.coef_)
        if pred_y is None:
            pred_y = clf.predict(self.vx)
        if probas is None:
            probas_ = clf.predict_proba(self.vx)[:,1]
        else:
            probas_ = probas
        fpr, tpr, thresholds = roc_curve(self.vy, probas_)
        failed = [float(nan) if self.vy[i] != self.vy[i] else 0 if pred_y[i] == self.vy[i] \
                             else 1 if pred_y[i] != self.vy[i] else float('nan') for i in range(len(pred_y))]
        roc_auc = auc(fpr, tpr)
        precision = precision_score(self.vy, pred_y)
        recall = recall_score(self.vy, pred_y)
        accuracy = accuracy_score(self.vy, pred_y)
        print(method, header+'_trans', roc_auc, 0, precision, recall, 0, 0, accuracy, 0)
        if plot_roc:
            with open('roc_objects_'+plot_prefix+'_'+method+'_'+header+'_trans.pyn', 'wb') as f:
                pickle.dump({'fpr':fpr, 'tpr':tpr, 'auc':roc_auc, 'failed':failed}, f)
        if self.test_array is not None:
            valid_array = np.array(
                [i for i in range(len(self.vy)) if i not in self.test_array])
            # print(max(self.test_array), max(valid_array), len(self.vy))
            # print(self.test_array)
            # print(valid_array)
            for name, array in zip(['test', 'valid'], [self.test_array, valid_array]):
                fpr, tpr, thresholds = roc_curve(
                    self.vy[array], probas_[array])
                roc_auc = auc(fpr, tpr)
                precision = precision_score(self.vy[array], pred_y[array])
                recall = recall_score(self.vy[array], pred_y[array])
                accuracy = accuracy_score(self.vy[array], pred_y[array])
                failed = [float(nan) if self.vy[i] != self.vy[i] else 0 if pred_y[i] == self.vy[i] \
                                    else 1 if pred_y[i] != self.vy[i] else float('nan') for i in array]
                print(method, header+'_'+name, roc_auc, 0,
                      precision, recall, 0, 0, accuracy, 0)
                if plot_roc:
                    with open('roc_objects_'+plot_prefix+'_'+method+'_'+header+'_trans_'+name+'.pyn', 'wb') as f:
                        pickle.dump({'fpr':fpr, 'tpr':tpr, 'auc':roc_auc, 'failed':failed}, f)
        kept_sampling = deepcopy(self.sample)
        if method == 'xgb':
            cv_results, est = self.cross_validation_xgb(clf, self.vx, self.vy)
        else:
            cv_results = cross_validate(
                clf, self.vx, self.vy, cv=self.sample, scoring=self.metrics, fit_params=None, return_estimator=True)
            est = cv_results['estimator']
        print(cv_results['test_roc_auc'])
        self.print_result(method, header, cv_results)
        if plot_roc:
            fpr, tpr, roc_auc, roc_auc_std, failed, answer = self.calc_mean_auc(clf, self.vx, self.vy, header, kept_sampling, est)
            with open('roc_objects_'+plot_prefix+'_'+method+'_'+header+'.pyn', 'wb') as f:
                pickle.dump({'fpr':fpr, 'tpr':tpr, 'auc':roc_auc, 'failed':failed, 'answer':answer, 'auc_std':roc_auc_std}, f)


    def calc_mean_auc(self, clf, x_matrix, y_vector, header, kept_sampling, clfs):
        global SEED
        from scipy import interp
        from sklearn.base import clone
        from joblib import Parallel, delayed
        from sklearn.model_selection import _validation
        from sklearn.metrics import check_scoring
        failed = [float('nan') for ty in y_vector]
        print(x_matrix.shape)
        sys.stdout.flush()
        mean_fpr = np.linspace(0, 1, 50)
        fprs, tprs, aucs = [], [], []
        best_fpr, best_tpr = [], []
        for j, (train, test) in enumerate(self.sample.split(x_matrix, y_vector)):
            clf.fit(x_matrix[train, :], y_vector[train])
            continue
        for j, (train, test) in enumerate(kept_sampling.split(x_matrix, y_vector)):
            # result, est = _validation._fit_and_score(clf, x_matrix, y_vector, scorers, train, test, verbose=0, return_train_score=False,
            #                                          return_estimator=True, error_score=np.nan, parameters=None, fit_params=None)

            fitted = clfs[j]
            probas_ = fitted.predict_proba(x_matrix[test,:])
            y_pred = fitted.predict(x_matrix[test,:])
            for h, x in enumerate(test):
                if y_vector[x] != y_vector[x]:
                    continue
                if y_vector[x] == y_pred[h]:
                    failed[x] = 0
                else:
                    failed[x] = 1
            fpr, tpr, thresholds = roc_curve(y_vector[test], probas_[:, 1])
            aucs.append(auc(fpr, tpr))
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        print(aucs)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        return mean_fpr, mean_tpr, np.mean(aucs), np.std(aucs), failed, [y*2 for y in y_vector]


    def plot_mean_auc(self, header, key, original_index, max_len):
        from scipy import interp
        from itertools import cycle
        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc
        global SEED
        methods = ['rf', 'lda', 'lasso', 'svm']
        failed = [[y*2 for y in self.Y]]
        print(self.X.shape)
        sys.stdout.flush()
        for i in range(len(methods)):
            cv = self.sample
            if methods[i] == 'svm':
                classifier = svm.SVC(kernel='rbf', probability=True, random_state=SEED, gamma='auto')
            elif methods[i] == 'rf':
                classifier = RandomForestClassifier(max_depth=4, random_state=SEED, n_estimators=500)
            elif methods[i] == 'lasso':
                classifier = linear_model.LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=SEED)
            else:
                classifier = LinearDiscriminantAnalysis()
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            fp_list, fn_list = [0.]*self.X.shape[0], [0.]*self.X.shape[0]
            tp_list, tn_list = [0.]*self.X.shape[0], [0.]*self.X.shape[0]
            for j, (train, test) in enumerate(cv.split(self.X, self.Y)):
                print(j, methods[i], '???')
                sys.stdout.flush()
                fitted = classifier.fit(self.X[train,:], self.Y[train])
                probas_ = fitted.predict_proba(self.X[test,:])
                y_pred = fitted.predict(self.X[test,:])
                for h, x in enumerate(test):
                    if self.Y[x] == y_pred[h]:
                        if self.Y[x] > 0:
                            tp_list[x] += 1
                        else:
                            tn_list[x] += 1
                    else:
                        if self.Y[x] > 0:
                            fn_list[x] += 1
                        else:
                            fp_list[x] += 1
                if np.min(probas_[:,1]) != np.min(probas_[:,1]):
                    print(np.min(probas_[:,1]), np.max(probas_[:,1]), probas_[:,0:10])
                    print(np.min(self.X[test,:]), np.max(self.X[test,:]))
                fpr, tpr, thresholds = roc_curve(self.Y[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='')
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Random', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right", fontsize=5)
            plt.savefig("auc_"+header+"_"+methods[i]+".pdf")
            plt.close()
            plt.clf()
            failed.append([fp_list[i]+fn_list[i] for i in range(len(fp_list))])
        print(failed)
        tfailed = [[float('nan') for i in range(max_len)] for f in failed]
        for i, n in enumerate(original_index):
            for j in range(len(failed)):
                tfailed[j][n] = failed[j][i]
        failed = tfailed
        fig, ax = plt.subplots()
        ax.imshow(failed)
        ax.set_yticklabels([key]+methods)
        ax.set_yticks(np.arange(len(methods)+1))
        ax.set_xticks(np.arange(0, len(failed[0]), 10))
        ax.set_aspect(5)
        plt.savefig('failure_'+header+'_'+'all'+'.pdf')
        plt.close(fig)
        plt.clf()


    def print_result(self, clf_name, header, cv_results):
        if self.auc:
            print(clf_name, header, cv_results['test_roc_auc'].mean(), cv_results['test_roc_auc'].std(),
                  cv_results['test_precision'].mean(), cv_results['test_recall'].mean(), cv_results['test_precision'].std(), cv_results['test_recall'].std(), cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std())
        else:
            print(clf_name, header, cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std(),
                  cv_results['test_precision'].mean(), cv_results['test_recall'].mean(), cv_results['test_precision'].std(), cv_results['test_recall'].std())


    def k_nearest_neighbors(self, header, k, plot_prefix=''):
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=k)
        try:
            if self.valid:
                neigh.fit(self.X, self.Y)
                self.test_validation_set('knn'+str(k), header, neigh, plot_prefix)
            else:
                cv_results = cross_validate(
                    neigh, self.X, self.Y, cv=self.sample, scoring=self.metrics)
                self.print_result('knn'+str(k), header, cv_results)
        except Exception as e:
            print(self.Y, self.vy)
            print(e)

    def multi_dim_classification_lda(self, header, plot_prefix=''):
        clf = LinearDiscriminantAnalysis()
        try:
            if self.valid:
                clf.fit(self.X, self.Y)
                self.test_validation_set('lda', header, clf, plot_prefix)
            else:
                cv_results = cross_validate(
                    clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
                self.print_result('lda', header, cv_results)
        except Exception as e:
            print(self.Y, self.vy)
            print(e)

    def multi_dim_classification_lasso(self, header, plot_prefix=''):
        global SEED
        try:
            for c in [1.0, 0.5, 0.1]:
                clf = linear_model.LogisticRegression(penalty='l1', C=c, solver='liblinear', random_state=SEED)
                if self.valid:
                    clf.fit(self.X, self.Y)
                    self.test_validation_set('lasso'+str(c), header, clf, plot_prefix)
                else:
                    cv_results = cross_validate(
                        clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
                    self.print_result('lasso'+str(c), header, cv_results)
        except Exception as e:
            print(self.Y, self.vy)
            print(e)

    def multi_dim_classification_svm(self, header, plot_prefix=''):
        global SEED
        clf = svm.SVC(kernel='rbf', probability=True, random_state=SEED, gamma='auto')
        try:
            if self.valid:
                clf.fit(self.X, self.Y)
                self.test_validation_set('svm-rbf', header, clf, plot_prefix)
            else:
                cv_results = cross_validate(
                    clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
                self.print_result('svm-rbf', header, cv_results)
        except Exception as e:
            print(self.Y, self.vy)
            print(e)

    def multi_dim_classification_imbalanced_svm(self, header):
        clf = svm.SVC(C=1.0, kernel='linear', class_weight='balanced')
        cv_results = cross_validate(
            clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
        self.print_result('imb_svm', header, cv_results)


    def multi_dim_classification_imbalanced_svm_sampling(self, header):
        clf = svm.SVC()
        prec_list, recall_list = [], []
        for X_train, X_test, Y_train, Y_test in self.sep_data():
            clf.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            precision, recall, _, _ = precision_recall_fscore_support(
                Y_test, Y_pred, pos_label=1, average='binary')
            prec_list.append(precision)
            recall_list.append(recall)
        self.print_result('imbs_svm', header, cv_results)

    def multi_dim_classification_rf(self, header, plot_prefix=''):
        global SEED
        clf = RandomForestClassifier(max_depth=5, random_state=SEED)
        try:
            if self.valid:
                clf.fit(self.X, self.Y)
                self.test_validation_set('rf', header, clf, plot_prefix)
            else:
                cv_results = cross_validate(
                    clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
                self.print_result('rf', header, cv_results)
        except Exception as e:
            print(self.Y, self.vy)
            print(e)


    def multi_dim_classification_ab(self, header, plot_prefix=''):
        global SEED
        clf = AdaBoostClassifier(n_estimators=100, random_state=SEED)
        try:
            if self.valid:
                clf.fit(self.X, self.Y)
                self.test_validation_set('ab', header, clf, plot_prefix)
            else:
                cv_results = cross_validate(
                    clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
                self.print_result('ab', header, cv_results)
        except Exception as e:
            print(self.Y, self.vy)
            print(e)

    def multi_dim_classification_xgb(self, header, plot_prefix=''):
        global SEED
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6
        }
        clf = xgb.XGBClassifier(**xgb_params)
        if self.valid:
            clf.fit(self.X, self.Y)
            y_pred_proba = clf.predict(self.vx)
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)
            self.test_validation_set('xgb', header, clf, plot_prefix, pred_y=y_pred, probas=y_pred_proba)
        else:
            cv_results, _ = self.cross_validation_xgb(clf, self.X, self.Y)
            self.print_result('xgb', header, cv_results)

    def cross_validation_xgb(self, clf, x, y):
        cv_results = {'test_roc_auc':[], 'test_precision':[], 'test_recall':[], 'test_accuracy':[]}
        clfs = []
        for X_train, X_test, Y_train, Y_test in self.sep_data(x, y):
            clf.fit(X_train, Y_train)
            y_pred_proba = clf.predict(X_test)
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)
            precision, recall, _, _ = precision_recall_fscore_support(
                Y_test, y_pred, pos_label=1, average='binary')
            auc = roc_auc_score(Y_test, y_pred_proba)
            accuracy = accuracy_score(Y_test, y_pred)
            cv_results['test_precision'].append(precision)
            cv_results['test_recall'].append(recall)
            cv_results['test_roc_auc'].append(auc)
            cv_results['test_accuracy'].append(accuracy)
            clfs.append(deepcopy(clf))
        for key in cv_results:
            cv_results[key] = np.array(cv_results[key])
        return cv_results, clfs

    def sep_data(self, x=None, y=None, cv=None):
        global SEED
        if cv is None:
            cv = self.CV
        skf = self.sample
        if x is None:
            x, y = self.X, self.Y
        for train_index, test_index in skf.split(x, y):
            X_train, X_test = np.take(x, train_index, axis=0), np.take(
                x, test_index, axis=0)
            Y_train, Y_test = np.take(
                y, train_index), np.take(y, test_index)
            yield X_train, X_test, Y_train, Y_test


    def comp_pvalue(self, X_train, Y_train):
        T, pvalue = ranksums([x for i, x in enumerate(X_train) if Y_train[i] == 1], [x for i, x in enumerate(X_train) if Y_train[i] == 0])
        return pvalue

    def cv_linear_disc(self, header, n_comp=50*4):
        print('Doing cv_linear')
        clf = LinearDiscriminantAnalysis()
        prec_list, recall_list, accuracy_list, auc_list = [], [], [], []
        for X_train, X_test, Y_train, Y_test in self.sep_data():
            pvalues = [ self.comp_pvalue(X_train[:,i], Y_train) for i in range(X_train.shape[1])]
            index = np.argsort(pvalues)
            index = np.array([j for i, j in enumerate(index) if i < n_comp])
            clf.fit(X_train[:,index], Y_train)
            Y_pred = clf.predict(X_test[:,index])
            precision, recall, _, _ = precision_recall_fscore_support(
                Y_test, Y_pred, pos_label=1, average='binary')
            prec_list.append(precision)
            recall_list.append(recall)
            accuracy = accuracy_score(Y_test, Y_pred)
            accuracy_list.append(accuracy)
            if self.auc:
                Y_pred = clf.predict_proba(X_test[:,index])
                auc = roc_auc_score(Y_test, Y_pred[:,1])
                auc_list.append(auc)
        if self.auc:
            print('cvlin', header, np.mean(auc_list), np.std(auc_list), np.mean(prec_list), np.mean(recall_list), np.std(prec_list), np.std(recall_list), np.mean(accuracy_list), np.std(accuracy_list))
        else:
            print('cvlin', header, np.mean(accuracy_list), np.std(accuracy_list), np.mean(
                prec_list), np.mean(recall_list))

    def raw_linear_disc(self, header, plot_prefix=''):
        clf = LinearDiscriminantAnalysis(n_components=200)
        if self.valid:
            clf.fit(self.X, self.Y)
            self.test_validation_set('alda', header, clf, plot_prefix)
        else:
            cv_results = cross_validate(
                clf, self.X, self.Y, cv=self.sample, scoring=self.metrics)
            self.print_result('alda', header, cv_results)


def MDS_scaling(samples, gname, n_comp=2):
    pos, similarities = comp_MDS(samples, n_comp)
    similarities = similarities.max() / similarities * 100
    similarities[np.isinf(similarities)] = 0
    return pos


def comp_MDS(samples, n_comp=2):
    global SEED
    from sklearn import manifold
    from sklearn.metrics import euclidean_distances
    print('# MDS...')
    similarities = euclidean_distances(samples)
    mds = manifold.MDS(n_comp, max_iter=5000, eps=1e-12, random_state=SEED,
                       n_init=10,
                       dissimilarity="precomputed", n_jobs=1, metric=False)
    pos = mds.fit_transform(similarities)
    return pos, similarities

def feature_clustering(samples, names, feature_name, gname):
    d = pd.DataFrame(data=samples, columns=feature_name, index=names)
    corr = d.corr()
    plot_dendrogram(d, gname, names)

def feature_label_map_valid(samples, names, ann, feature_name, vsamples, vnames, vann, prefix=''):
    iann = np.hstack((ann, vann))
    feature_label_map(np.vstack((samples, vsamples)), names+vnames, iann, feature_name, prefix=prefix+'_valid_')

def feature_label_map(samples, names, ann, feature_name, prefix=''):
    global FEATURE
    import pandas as pd
    d = pd.DataFrame(data=samples, columns=feature_name, index=names)
    print(prefix)
    for i, key in enumerate(FEATURE):
        if key not in ["MGMT", "GBM", "MGMT GBM"]:
            continue
        normd = normalize_row_col_mat(d.loc[:,:])
        print(np.min(normd), np.max(normd))
        s = 30
        index_set = []
        max_ite = (2 if 'valid' in prefix else 1)
        print(ann[i,:].shape, len(names))
        for h in range(max_ite):
            if h == 0:
                index, marker, lprefix = [i for i in range(len(ann[i,:])) if 'tcia' not in names[i]], 'o', 'NCC'
            else:
                index, marker, lprefix = [i for i in range(len(ann[i,:])) if 'tcia' in names[i]], 'x', 'TCIA'
            if key == 'GBM':
                cmap = cm.get_cmap('viridis')
                color_list = ['red', 'black', 'red', 'black']
                if h == 0:
                    index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 1 and j in index], 'color':color_list[0:1], 'marker':marker, 'label': str(lprefix+'_'+'+')})
                    index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 0 and j in index], 'color':color_list[1:2], 'marker':marker, 'label':str(lprefix+'_'+'-')})
                else:
                    index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 1 and j in index], 'color':color_list[2:3], 'marker':'+', 'label': str(lprefix+'_'+'+')})
                    index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 0 and j in index], 'color':color_list[3:4], 'marker':'x', 'label':str(lprefix+'_'+'-')})
            else:
                index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 1 and j in index], 'color':'red', 'marker':marker, 'label': str(lprefix+'_'+'+')})
                index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 0 and j in index], 'color':'black', 'marker':marker, 'label':str(lprefix+'_'+'-')})
                if key in ['16CpG', 'CpG', 'TERT', 'MGMT']:
                    index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 1 and ann[0,j] == 0 and j in index], 'color':'pink', 'marker':marker, 'label': str(lprefix+'_LrGG_'+'+')})
                    index_set.append({'data':[j for j, a in enumerate(ann[i,:]) if a == 0 and ann[0,j] == 0 and j in index], 'color':'gray', 'marker':marker, 'label': str(lprefix+'_LrGG_'+'-')})
        for l in [100, 200, 500, 1000]:
            tsne = TSNE(n_components=2, learning_rate=l).fit_transform(normalize_row_col_mat(d.loc[:, :]))
            fig, ax = plt.subplots()
            for index in index_set:
                if key == 'GBM':
                    print(index['color'])
                    plt.scatter(tsne[index['data'], 0], tsne[index['data'], 1], color=index['color'], s=s, lw=1, label=index['label'], marker=index['marker'])
                else:
                    plt.scatter(tsne[index['data'], 0], tsne[index['data'], 1], color=index['color'], s=s, lw=1, label=index['label'], marker=index['marker'])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4)
            plt.tight_layout()
            plt.savefig(('tsne_' + prefix + key + '_' + str(l) + '.pdf').replace('/', ''))
            plt.close(fig)
            plt.clf()
        x = StandardScaler().fit_transform(remove_nan(d.loc[:,:]))
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(x)
        for index in index_set:
            if key == 'GBM':
                plt.scatter(pca_data[index['data'], 0], pca_data[index['data'], 1], color=index['color'] , s=s, lw=1, label=index['label'], marker=index['marker'])
            else:
                plt.scatter(pca_data[index['data'], 0], pca_data[index['data'], 1], color=index['color'], s=s, lw=1, label=index['label'], marker=index['marker'])
        plt.legend(scatterpoints=1, loc=1, shadow=False)
        plt.tight_layout()
        plt.savefig(('pca_' + prefix + key + '_' + str(l) + '.pdf').replace('/', ''))
        plt.close()
        plt.clf()
        sys.stdout.flush()

def get_CV(key, gbm=False):
    global CV
    if gbm and key != 0: # Not MGMT GBM
        return 2
    if key in ['1p19q', 'Group A', 'Group B', 'Group C', 'Group D', 'IDH GBM', 'TERT GBM', 'TERT LGG', 'IDH LGG']:
        return math.ceil(CV/2)
    else:
        return 5

def check_not_enough_positives(data, cv, header=''):
    positive, negative = len([y for y in data if y == 1]), len([y for y in data if y == 0])
    if len(data) == 0 or positive < cv or negative < cv:
        print('No enough data', header, positive, negative, cv)
        return True
    else:
        return False


def supervised_classification(samples, label, key, feature_name, header, vx=None, vy=None, auc=False, plot_prefix='', divide=False, gbm=False):
    global FEATURE
    print('# classification...')
    print(key, label.shape)
    original_index = np.where(label == label)[0]
    max_len = len(label)
    print('Data and annotation dimension:', [x for x in np.unique(label) if x == x], len(label), samples.shape)
    sys.stdout.flush()
    if len(original_index) == 0:
        return
    sup = SupervisedClassification(samples, label, auc=auc, CV=get_CV(key, gbm))
    if vx is not None and vy is not None:
        sup.set_valid_dataset(vx.real, vy)
    if sup.valid:
        if check_not_enough_positives(sup.vy, get_CV(key, gbm), key): return
    else:
        if check_not_enough_positives(sup.Y, get_CV(key, gbm), key): return
    if divide:
        sup.divide_test_set(key)
    if header.split('_')[-1] in ['test', 'validation']:
        sup.plot_mean_auc(header+plot_prefix, key, original_index, max_len)
    else:
        print('no plot', header.split('_')[-1])
    for i in range(1, 6, 2):
        sup.k_nearest_neighbors(header, i, plot_prefix)
        sys.stdout.flush()
    sup.multi_dim_classification_xgb(header, plot_prefix)
    sys.stdout.flush()
    sup.multi_dim_classification_lda(header, plot_prefix)
    sys.stdout.flush()
    sup.multi_dim_classification_rf(header, plot_prefix)
    sys.stdout.flush()
    sup.multi_dim_classification_ab(header, plot_prefix)
    sys.stdout.flush()
    sup.multi_dim_classification_svm(header, plot_prefix)
    sys.stdout.flush()

def feature_filtering(data, samples, label, key, feature_name, header, vx=None, vy=None, auc=False, plot_prefix='', divide=False, gbm=False):
    print('# classification...')
    print(samples.shape)
    original_index = np.where(label == label)[0]
    max_len = len(label)
    print(np.unique(label), len(label))
    print(np.where(label == label))
    sys.stdout.flush()
    samples = samples[original_index,:]
    label = [label[i] for i in original_index]
    print(samples.shape)
    sup = SupervisedClassification(samples, label, auc=auc, CV=get_CV(key, gbm))
    if vx is not None and vy is not None:
        sup.set_valid_dataset(vx, vy)
    if sup.valid:
        if check_not_enough_positives(sup.vy, get_CV(key, gbm), key): return
    else:
        if check_not_enough_positives(sup.Y, get_CV(key, gbm), key): return
    if divide:
        sup.divide_test_set(feature_name)
    if header.split('_')[-1] in ['test', 'validation']:
        sup.plot_mean_auc(header+plot_prefix, key, original_index, max_len)
        sup.cv_linear_disc(header)
    sup.raw_linear_disc(header, plot_prefix)
    sup.multi_dim_classification_lasso(header, plot_prefix)


class Dataset:
    def __init__(self, arg):
        self.arg = arg
        self.afname = arg.ann
        self.feat_ext = FeatureExtrcation(self.arg)
        self.patient_list = [self.feat_ext.get_patient_list(fname) for fname in self.arg.pfile]
        self.feature_data, self.feature_group, self.feature_names, self.grade_list =  None, None, None, None
        self.feature_lib = FeatureLibrary()
        self.valid_data, self.valid_list, self.valid_annotations = None, None, None
        self.full_annotations, self.annotation_names = None, None
        self.temp_ann_feature = dict([('feature', []), ('names', []), ('group', []), ('valid_feature', [])])

    def remove_index(self, index, included=False):
        if included:
            self.feature_data = self.feature_data[:,index]
            self.feature_names = [self.feature_names[i] for i in index]
            self.feature_group = [self.feature_group[i] for i in index]
        else:
            self.feature_data = np.delete(self.feature_data, index, 1)
            self.feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if i not in index]
            self.feature_group = [self.feature_group[i] for i in range(len(self.feature_group)) if i not in index]

    def remove_incomplete_data(self):
        index = np.array([i for i in range(self.feature_data.shape[0]) if not np.isnan(self.feature_data[i,:]).any()])
        self.feature_data = self.feature_data[index,:]
        self.grade_list = [self.grade_list[i] for i in index]
        self.full_annotations = self.full_annotations[:,index]

    def plot_feature_relationship(self, output):
        for i in range(-1, max(self.feature_group)):
            if i < 0:
                index = [j for j in range(self.feature_data.shape[1])]
            else:
                index = [j for j in range(self.feature_data.shape[1]) if self.feature_group[j] == i]
            d = pd.DataFrame(data=self.feature_data[:,index])
            corr = d.corr()
            corr = corr.values
            g = sns.clustermap(corr, row_cluster=False, col_cluster=False)
            plt.savefig('corr_'+str(i)+output)
            plt.close()
            plt.clf()
            g = sns.clustermap(corr, row_cluster=True, col_cluster=True, metric='chebyshev')
            plt.savefig('cluster_'+str(i)+output)
            plt.close()
            plt.clf()

    def apply_filtering(self, prefix):
        index = [i for i, s in enumerate(np.sum(self.feature_data, axis=1)) if s == 0.]
        index = list(set(index+[i for i, s in enumerate(np.var(self.feature_data, axis=1)) if s == 0.]))
        self.remove_index(index)
        index = remove_colinear(remove_nan(self.feature_data))
        self.remove_index(index, True)
        self.feature_data = histogram_normalization(self.feature_data)
        print(self.feature_data.shape)
        print(len(self.feature_names))
        print(len(self.feature_group))

    def write_dataset(self, prefix):
        feature_data, grade_list, feature_names, feature_group, feature_lib = self.feat_ext.get_feature_data(self.patient_list[0])
        with open('full_features'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(feature_data, f)
        with open('feature_group'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(feature_group, f)
        with open('feature_names'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(feature_names, f)
        with open('grade_list'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(grade_list, f)
        with open('feature_lib'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(feature_lib, f)

    def write_valid_dataset(self, prefix):
        with open('full_features'+prefix+'.pyn', mode='rb') as f:
            self.feature_data = CustomUnpickler(f).load()
        with open('feature_group'+prefix+'.pyn', mode='rb') as f:
            self.feature_group = CustomUnpickler(f).load()
        with open('feature_names'+prefix+'.pyn', mode='rb') as f:
            self.feature_names = CustomUnpickler(f).load()
        with open('grade_list'+prefix+'.pyn', mode='rb') as f:
            self.grade_list = CustomUnpickler(f).load()
        with open('feature_lib'+prefix+'.pyn', mode='rb') as f:
            self.feature_lib = CustomUnpickler(f).load()
        valid_data, valid_list, _, _, _ = self.feat_ext.get_feature_data(self.patient_list[1], validation=True, feature_names=self.feature_names, feature_lib=self.feature_lib)
        with open('valid_features'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(valid_data, f)
        with open('valid_list'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(valid_list, f)

    def write_reduced_dataset(self, prefix):
        self.apply_filtering(prefix)
        with open('red_features'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(self.feature_data, f)
        with open('red_feature_group'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(self.feature_group, f)
        with open('red_feature_names'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(self.feature_names, f)

    def write_annotations(self, prefix):
        with open('grade_list'+prefix+'.pyn', mode='rb') as f:
            self.grade_list = CustomUnpickler(f).load()
        with open('feature_lib'+prefix+'.pyn', mode='rb') as f:
            self.feature_lib = CustomUnpickler(f).load()
        if os.path.exists('valid_list'+prefix+'.pyn'):
            with open('valid_list'+prefix+'.pyn', mode='rb') as f:
                self.valid_list = CustomUnpickler(f).load()
        ann = Annotation(self.grade_list, self.valid_list)
        full_annotations, annotation_names = ann.get_positive_negative_matrix(self.afname)
        valid_annotations, _ = ann.get_positive_negative_matrix(self.afname, self.valid_list)
        with open('annotation_names'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(annotation_names, f)
        with open('full_annotations'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(full_annotations, f)
        with open('valid_annotations'+prefix+'.pyn', mode='wb') as f:
            pickle.dump(valid_annotations, f)

    def load_dataset(self, ann_feature, filtering, post=False):
        self.load_dataset_all(filtering)
        if len(self.arg.ann_feature) > 0:
            self.load_feature_ann(post)

    def load_dataset_all(self, filtering, verbose=True):
        global FEATURE
        prefix = self.arg.output
        if filtering and False:
            with open('red_features'+prefix+'.pyn', mode='rb') as f:
                self.feature_data = CustomUnpickler(f).load()
            with open('red_feature_group'+prefix+'.pyn', mode='rb') as f:
                self.feature_group = CustomUnpickler(f).load()
            with open('red_feature_names'+prefix+'.pyn', mode='rb') as f:
                self.feature_names = CustomUnpickler(f).load()
        else:
            with open('full_features'+prefix+'.pyn', mode='rb') as f:
                self.feature_data = CustomUnpickler(f).load()
            with open('feature_group'+prefix+'.pyn', mode='rb') as f:
                self.feature_group = CustomUnpickler(f).load()
            with open('feature_names'+prefix+'.pyn', mode='rb') as f:
                self.feature_names = CustomUnpickler(f).load()
            if len(self.arg.pfile) > 1:
                with open('valid_features'+prefix+'.pyn', mode='rb') as f:
                    self.valid_data = CustomUnpickler(f).load()
                with open('valid_list'+prefix+'.pyn', mode='rb') as f:
                    self.valid_list = CustomUnpickler(f).load()
                with open('valid_annotations'+prefix+'.pyn', mode='rb') as f:
                    self.valid_annotations = CustomUnpickler(f).load()
        with open('grade_list'+prefix+'.pyn', mode='rb') as f:
            self.grade_list = CustomUnpickler(f).load()
        with open('feature_lib'+prefix+'.pyn', mode='rb') as f:
            self.feature_lib = CustomUnpickler(f).load()
        with open('full_annotations'+prefix+'.pyn', mode='rb') as f:
            self.full_annotations = CustomUnpickler(f).load()
        with open('annotation_names'+prefix+'.pyn', mode='rb') as f:
            self.annotation_names = CustomUnpickler(f).load()
        print(self.annotation_names)
        print(self.full_annotations.shape)
        for key in range(len(FEATURE)):
            for h, d in zip(['training', 'validation'], [self.full_annotations, self.valid_annotations]):
                if d is None: continue
                print(key, d.shape)
                vec = d[key]
                print(h, key, FEATURE[key], len([1 for v in vec if v == 1]), len([0 for v in vec if v == 0]))
                print(h, key, FEATURE[key], len([1 for i in range(len(vec)) if vec[i] == 1 and d[0,i] == 1]), len([0 for i in range(len(vec)) if vec[i] == 0 and d[0,i] == 1]))


    def load_feature_ann(self, post):
        ann = Annotation(self.grade_list, self.valid_list)
        if len(self.arg.ann_feature) > 0:
            group_index = self.feature_lib.insert(['annotation', 'all'])
            part_names, part_features, vpart_features = ann.get_annotation_feature(self.arg.ann, self.arg.ann_feature)
            part_group = [group_index]*len(part_names)
        else:
            part_names, part_features, vpart_features = [], [], []
            part_group = []
        with open("part_features.pyn", "wb") as f:
            pickle.dump(part_features, f)
        with open("vpart_features.pyn", "wb") as f:
            pickle.dump(vpart_features, f)
        if not post:
            self.feature_data = np.column_stack((self.feature_data, part_features))
            self.feature_names = self.feature_names+part_names
            self.feature_group = self.feature_group+part_group
            if vpart_features is not None:
                self.valid_data = np.column_stack((self.valid_data, vpart_features))
        else: #store to temporary list
            self.temp_ann_feature = dict([('feature', part_features), ('names', part_names), ('group', part_group)])
            if vpart_features is not None:
                self.temp_ann_feature['valid_feature'] = vpart_features



    def get_cindex(self, sp=-1, filt=''):
        cindex = [int(i) for i in range(len(self.feature_group)) if self.feature_group[i] == sp or sp < 0]
        return cindex

    def search_suitable_features(self, tags):
        global UNIQ_METHODS
        index = []
        for i, d in enumerate(self.feature_lib.dicts):
            for j, tag in enumerate(tags):
                if tag not in d:
                    break
                if j == len(tags)-1:
                    index.append(i)
        return index

    def get_cindex_tag_based(self, sp=None, filt=''):
        # ex sp == None -> all
        # ex sp == ['GD', 'resnet']
        cindex = []
        if sp is None or sp == 'all':
            cindex = [int(i) for i in range(len(self.feature_group))]
        else:
            assert type(sp) == list
            if 'all' in sp:
                if len(sp) == 1:
                    cindex = [int(i) for i in range(len(self.feature_group))]
                else:
                    sp.remove('all')
            if len(cindex) == 0:
                sp_list = self.search_suitable_features(sp)
                cindex = [int(i) for i in range(len(self.feature_group)) if self.feature_group[i] in sp_list]
        if len(filt) > 0:
            cindex = [int(i) for i in cindex if not re.match('^tumor_', self.feature_name[i])]
            cindex = [int(i) for i in cindex if not re.match('^size_', self.feature_name[i])]
            cindex = [int(i) for i in cindex if not re.match('^centroid_', self.feature_name[i])]
        return cindex

    def fet_cindex(self, ann, cindex):
        global AUROC
        annotations = self.full_annotations[ann,:]
        T, N = len([x for x in annotations if x == 1]), len([x for x in annotations if x == 0])
        if AUROC:
            min_pvalues = [comp_auroc(self.feature_data[:,c], annotations, T, N, c, self.feature_names[c], i%100==0) for i, c in enumerate(cindex)]
            min_pvalues = [1.-x if x < 0.5 else x for x in min_pvalues]
        else:
            min_pvalues = [comp_pvalues(self.feature_data[:,c], annotations, T, N, c, self.feature_names[c], i%100==0) for i, c in enumerate(cindex)]
        pvalue_filtered = [(i, cindex[i]) for i in np.argsort(min_pvalues)]
        if AUROC:
            pvalue_filtered = pvalue_filtered[::-1]
        print(pvalue_filtered[0:10])
        return [i for i, c in pvalue_filtered], min_pvalues, AUROC


def plot_corr_annotations(ann, prefix=''):
    global FEATURE
    ann = remove_nan(ann)
    d = pd.DataFrame(ann, index=FEATURE)
    corr = d.transpose().corr()
    fig, ax = plt.subplots()
    plt.imshow(corr)
    ax.grid(False)
    ax.set_yticks(np.arange(0, len(FEATURE)))
    ax.set_xticks(np.arange(0, len(FEATURE)))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('ann_corr_' + prefix + '_sample.pdf')
    plt.close('all')
    plt.clf()


def feature_and_ann_clustering(data, ann_flag=True):
    for i in range(max(data.feature_group)):
        print(i, list(data.feature_group).count(i))
    for sp in range(max(data.feature_group), -2, -1):
    # for sp in range(-1, -2, -1):
        cindex = data.get_cindex(sp)
        print('Plot clustering', 'feature type', sp, 'feature dimension', len(cindex))
        if sp == max(data.feature_group): # Annotation feature
            sample_order = plot_corr_features(remove_nan(data.feature_data[:,np.array(cindex)]), [data.feature_names[i] for i in cindex], data.grade_list, data.arg.output+'_'+str(sp))
        else:
            sample_order = plot_corr_features(normalize_mat(data.feature_data[:,np.array(cindex)]), [data.feature_names[i] for i in cindex], data.grade_list, data.arg.output+'_'+str(sp))
        plot_ann_clustering([data.grade_list[int(i)] for i in sample_order], data.full_annotations, data.annotation_names, data.grade_list, data.arg.output+'_'+str(sp))
    plt.close('all')
    if data.valid_data is not None:
        cindex = data.get_cindex(-1)
        print('valid', data.feature_data.shape, data.valid_data.shape, len(data.valid_list))
        sample_order = plot_corr_features(normalize_mat(data.valid_data[:,np.array(cindex)]), [data.feature_names[i] for i in cindex], data.valid_list, data.arg.output+'_valid_'+str(sp))
        plot_ann_clustering([data.valid_list[int(i)] for i in sample_order], data.valid_annotations, data.annotation_names, data.valid_list, data.arg.output+'_valid_'+str(sp))
        plot_corr_annotations(data.valid_annotations, prefix=data.arg.output+'_valid_')
        feature_label_map_valid(data.feature_data, data.grade_list, data.full_annotations, data.feature_names, data.valid_data, data.valid_list, data.valid_annotations, prefix=data.arg.output)
    # plot_corr_annotations(data.full_annotations)
    print(data.full_annotations)
    # sample_order = plot_corr_features(data.full_annotations.transpose(), data.annotation_names, data.grade_list, 'ann')
    feature_label_map(data.feature_data, data.grade_list, data.full_annotations, data.feature_names, prefix=data.arg.output)

def comp_auroc(features, annotations, T, N, c, name, verbose=False):
    index = [i for i in range(len(annotations)) if annotations[i] == annotations[i]]
    tann, tfea = [annotations[i] for i in index], features[np.array(index)]
    fpr, tpr, thresholds = roc_curve(tann, remove_nan(tfea), pos_label=1)
    auroc = auc(fpr, tpr)
    if verbose:
        print("#", c, name, auroc)
        sys.stdout.flush()
    return auroc

def comp_pvalues(features, annotations, T, N, c, name, verbose=False):
    tables = [[0, 0], [T, N]]
    order = np.argsort(features)
    pvalues = []
    flag = True
    for i in range(len(order)):
        p = annotations[order[i]]
        if p == 1:
            tables[0][0] += 1
            tables[1][0] -= 1
        elif p == 0:
            tables[0][1] += 1
            tables[1][1] -= 1
        if i == len(order)-1 or features[order[i]] != features[order[i+1]]:
            oddsratio, pvalue = stats.fisher_exact(tables)
            if flag and pvalue < 0.01:
                if verbose:
                    print("#", c, name, tables)
                flag = False
            pvalues.append(pvalue)
    sys.stdout.flush()
    return min(pvalues)

def apply_pca_to_other(all_pos, emb):
    # try:
    return emb.transform(all_pos)
    # except Exception as e:
    #     return np.zeros(shape=(all_pos.shape[0], emb.n_components))

def compress_PCA(all_pos, n_comp, pca_add, min_dim):
    global SEED
    from sklearn.decomposition import PCA
    print('before', all_pos.shape)
    embedding = PCA(n_components=min(min_dim, n_comp-pca_add), whiten=True, random_state=SEED)
    embedding.fit(all_pos)
    all_pos = embedding.transform(all_pos)
    print('after', all_pos.shape)
    return all_pos, embedding

def apply_nmf_to_other(all_pos, W):
    # try:
    return np.matmul(all_pos, scipy.linalg.pinv(W))
    # except Exception as e:
        # return np.zeros(shape=(all_pos.shape[0], W.shape[1]))

def compress_nimfa(all_pos, n_comp, nmf_add):
    import nimfa
    print('before', all_pos.shape)
    nmf = nimfa.Nmf(all_pos.clip(min=0), seed='nndsvd', rank=min(min(all_pos.shape), n_comp-nmf_add), max_iter=200) # before 100
    nmf_fit = nmf() # V=HW
    H = nmf_fit.basis() #URL ・ 特徴の行列
    W = nmf_fit.coef() #特徴　・　要素の行列
    all_pos = H
    print('after', all_pos.shape)
    # E = np.linalg.norm(nmf.residuals())
    return H, W

def compress_nmf(all_pos, n_comp):
    from sklearn.decomposition import NMF
    global SEED
    model = NMF(n_components=n_comp*4, init='random', random_state=SEED)
    print('fit NMF')
    sys.stdout.flush()
    W = model.fit_transform(all_pos.clip(min=0))
    H = model.components_
    return W

def compress_snmf(all_pos, n_comp):
    import nimfa
    print('before', all_pos.shape)

    nmf = nimfa.Snmf(all_pos.clip(min=0), seed="random_vcol", rank=min(min(all_pos.shape), n_comp*4), max_iter=12, version='l',
          eta=1., beta=1e-4, i_conv=10, w_min_change=0)
    nmf_fit = nmf()
    H = nmf_fit.basis() #URL ・ 特徴の行列
    W = nmf_fit.coef().transpose() #特徴　・　要素の行列
    print('after', H.shape)
    return H


class DimensionReduction:
    def __init__(self, data, type='', name=None, data_list=None, feat_list=None, feat_names=None):
        self.data = data
        self.name, self.matrix, self.annotation, self.feat_names = self.set_matrix(type, name, data_list, feat_list, feat_names)

    def set_matrix(self, type, name, data_list, feat_list, feat_names):
        if name is None:
            if self.data.valid_data is not None:
                if type in ['both', 'oppose']:
                    return ['test', 'validation'], [self.data.feature_data, self.data.valid_data], [self.data.full_annotations, self.data.valid_annotations], self.data.annotation_names
                elif type == 'validation':
                    return ['validation'], [self.data.valid_data], [self.data.full_annotations], self.data.annotation_names
            return ['test'], [self.data.feature_data], [self.data.full_annotations], self.data.annotation_names
        else:
            return name, data_list, feat_list, feat_names

    def plot_reduced_data(self, header, mat, cohort):
        colors = ['red', 'black', 'blue']
        cdict = dict([(x, colors[i]) for i, x in enumerate(list(set(cohort)))])
        for i, c in enumerate(set(cohort)):
            index = [j for j in range(len(cohort)) if cohort[j] == c]
            plt.scatter(mat[index,0], mat[index,1], color=colors[i])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig('red_cohort_'+header+'.pdf')
        plt.close()
        plt.clf()
        for i, x in enumerate(self.feat_names):
            if x == 'GBM' or x == 'MGMT':
                for j, c in enumerate(set(cohort)):
                    index = [h for h, y in enumerate(self.annotation[j][i,:]) if y == 0]
                    plt.scatter(mat[index,0], mat[index,1], color='black', alpha=[0.2, 1.][j])
                    index = [h for h, y in enumerate(self.annotation[j][i,:]) if y > 0]
                    plt.scatter(mat[index,0], mat[index,1], color='red', alpha=[0.2, 1.][j])
                plt.savefig('red_'+x+'_'+header+'.pdf')
                plt.close()
                plt.clf()

    def compress_each_data_concat(self, image, method, sp, n_comp, filt, each=True, plot=False, post_features=0):
        print('compress_each_data_concat')
        if each:
            cindex = self.data.get_cindex_tag_based([sp, image], filt)
            dim = n_comp
        else:
            cindex = self.data.get_cindex_tag_based([sp], filt)
            dim = n_comp*4
        if len(cindex) == 0:
            return [None]
        if method == 'NMF': #apply to training and test data, then multiply inverse W (dictionary) to validation
            feature_matrix = self.matrix[0]
            feature_matrix = remove_nan(feature_matrix[:,np.array(cindex, dtype=np.int32)])
            pos, W = compress_nimfa(feature_matrix, dim, post_features)
            print('# NMF_compressed', pos.shape, W.shape)
            red = [pos]
            for x in self.matrix[1:]:
                red.append(apply_nmf_to_other(remove_nan(x[:,np.array(cindex, dtype=np.int32)]), W))
        elif method == 'PCA' and len(self.matrix) > 1:
            feature_matrix = self.matrix[0]
            feature_matrix = remove_nan(feature_matrix[:,np.array(cindex, dtype=np.int32)])
            pos, emb = compress_PCA(feature_matrix, dim, post_features, min([min(x.shape) for x in self.matrix]))
            print('# PCA_compressed', pos.shape)
            red = [pos]
            for x in self.matrix[1:]:
                print(feature_matrix.shape, x.shape)
                red.append(apply_pca_to_other(remove_nan(x[:,np.array(cindex, dtype=np.int32)]), emb))
        else:
            feature_matrix = (self.matrix[0] if len(self.matrix) == 1 else np.concatenate([x for x in self.matrix]))
            feature_matrix = remove_nan(feature_matrix[:,np.array(cindex, dtype=np.int32)])
            if method == 'PCA':
                pos, _ = compress_PCA(feature_matrix, dim, post_features, min(feature_matrix.shape))
            elif method == 'MDS':
                pos = MDS_scaling(feature_matrix, str(n_comp), min(min(feature_matrix.shape), dim)-post_features)
            else:
                pos = feature_matrix
            rows = [0]+[x.shape[0] for x in self.matrix]
            print('divide_rows_into_each_cohort', rows)
            if plot:
                self.plot_reduced_data(image+'_'+method+'_'+sp+'_'+str(n_comp)+self.data.arg.output, pos, [self.name[i] for i in range(len(self.name)) for j in range(self.matrix[i].shape[0])])
            red = [pos[sum(rows[0:i]):(sum(rows[0:(i+1)])),:] for i in range(1, len(rows))]
            assert all([rows[i+1] == len(red[i]) for i in range(len(red))])
        return red

def get_reduced_mat(method, data, sp, n_comp, filt, validation, each=False):
    dimr = DimensionReduction(data, type=validation)
    if not each:
        print('dimension reduction (all):', method)
        all_pos = dimr.compress_each_data_concat('', method, sp, n_comp, filt, each=each, plot=True, post_features=len(data.temp_ann_feature['names']))
    else:
        print('dimension reduction (each weighting):', method)
        all_pos = []
        for m, image in enumerate(['GD', 'T1', 'T2', 'FLAIR']):
            # if (sp == 'annotation' or sp == 'tissue') and m > 0: # only use GD
            #     break
            pos = dimr.compress_each_data_concat(image, method, sp, n_comp, filt, each=each, plot=False)
            if pos is None:
                continue
            if m == 0:
                all_pos = [x.copy() for x in pos]
            else:
                all_pos = [np.column_stack((all_pos[i], pos[i])) for i in range(len(pos))]
    print('reduced mat', sp)
    print('cohort size', len(all_pos), 'feature dimension', all_pos[0].shape)
    return all_pos


def add_feature_ann(data, all_pos, feat_names, validation):
    dname = ['feature', 'valid_feature']
    if validation == 'validation':
        assert len(all_pos) == 1 or all_pos[0].shape[0] == 0
        dname = ['valid_feature']
    elif validation == 'test':
        dname = ['feature']
    print(data.temp_ann_feature['feature'])
    if type(data.temp_ann_feature['feature']) == list: # No annotation
        return all_pos, feat_names
    for i in range(min(len(dname), len(all_pos))):
        if all_pos[i].shape[0] == 0:
            all_pos[i] = data.temp_ann_feature[dname[i]]
        else:
            print('add_feature_ann', all_pos[i].shape, data.temp_ann_feature[dname[i]].shape)
            all_pos[i] = np.column_stack((all_pos[i], data.temp_ann_feature[dname[i]]))
        if i == 0:
            feat_names[i] = feat_names[i]+data.temp_ann_feature['names']
    return all_pos, feat_names

def get_n_comp_list(method, arg):
    if arg.dim_list != '':
        return list(map(int, arg.dim_list.split(',')))
    else:
        if method == 'MDS':
            return [50, 2, 5, 10, 100]
        else:
            return [50, 2, 5, 10]

def get_complete_matrix_data(method, data, arg, validation, n_comp, sp, filt, n_comp_list):
    cindex = data.get_cindex_tag_based([sp], '')
    if sp == 'annotation' and arg.post:
        all_pos, feat_names = ([np.zeros(shape=(0, 0)), np.zeros(shape=(0, 0))] if validation else [[]]), ([[], []] if validation else [[]])
    else:
        # if method == 'PCA' and (sp == 'all' or (type(sp) == int and sp == -1)):
        #     all_pos, feat_names = get_matrix_data(method, data, validation, n_comp, sp, filt, n_comp_list=n_comp_list, post=arg.post)
        # else:
        all_pos, feat_names = get_matrix_data(method, data, validation, n_comp, sp, filt, post=arg.post)
    if arg.post:
        all_pos, feat_names = add_feature_ann(data, all_pos, feat_names, validation)
    return all_pos, feat_names

def matrix_normalization(all_pos, scale_flag=False):
    global SCALE
    if scale_flag and SCALE:
        print('Scale applying...')
        all_pos[0] = StandardScaler().fit_transform(remove_nan(all_pos[0], max_val=1e5).real)
        if len(all_pos) > 1:
            all_pos[1] = StandardScaler().fit_transform(remove_nan(all_pos[1], max_val=1e5))
    else:
        all_pos[0] = remove_nan(all_pos[0]).real
        if len(all_pos) > 1:
            all_pos[1] = remove_nan(all_pos[1])
    return all_pos

def dimension_reduction(method, data, arg):
    global FEATURE, VALIDATION
    n_comp_list = get_n_comp_list(method, arg)
    for validation in VALIDATION:
        if data.valid_data is None and validation != 'test':  continue
        for sp in get_sp_range():
            for n_comp in n_comp_list:
                if arg.ann_pred and sp != 'annotation': continue
                if not arg.ann_pred and sp != 'all': continue
                for filt in ['']:
                    if len(filt) > 0 and sp != 'basic':
                        continue
                    print(n_comp, sp, 'filt:', filt, validation)
                    all_pos, feat_names = get_complete_matrix_data(method, data, arg, validation, n_comp, sp, filt, n_comp_list)
                    sys.stdout.flush()
                    if all_pos[0] is None:
                        print('No features', sp)
                        continue
                    all_pos = matrix_normalization(all_pos, (not arg.ann_pred and len(arg.ann_feature) > 0))
                    for ann in range(len(FEATURE)):
                        print(len(all_pos[0]))
                        print(all_pos[0].shape, n_comp, len(all_pos))
                        header = method+'_' + str(n_comp) + '_' + str(sp) + '_' + FEATURE[ann] + filt + '_' + validation
                        plot_prefix = arg.output+('_post' if arg.post else '')
                        if validation == 'both':
                            supervised_classification(all_pos[0], data.full_annotations[ann,:], FEATURE[ann], feat_names,
                                                      header, vx=all_pos[1], vy=data.valid_annotations[ann,:], auc=arg.auc, plot_prefix=plot_prefix, gbm=arg.gbm)
                        elif validation == 'oppose':
                            supervised_classification(all_pos[1], data.valid_annotations[ann,:], FEATURE[ann], feat_names,
                                                      header, vx=all_pos[0], vy=data.full_annotations[ann,:], auc=arg.auc, plot_prefix=plot_prefix, divide=True, gbm=arg.gbm)
                        elif validation == 'validation':
                            supervised_classification(all_pos[0], data.valid_annotations[ann,:], FEATURE[ann],feat_names, header, auc=arg.auc, plot_prefix=plot_prefix, gbm=arg.gbm)
                        else:
                            supervised_classification(all_pos[0], data.full_annotations[ann,:], FEATURE[ann], feat_names, header, auc=arg.auc, plot_prefix=plot_prefix, gbm=arg.gbm)
                        sys.stdout.flush()
                if len(data.get_cindex_tag_based([sp], '')) <= n_comp:
                    break


def get_matrix_data(method, data, validation, n_comp, sp, filt, n_comp_list=[], post=False):
    temp = method+'_'+validation+'_'+str(n_comp)+'_'+str(sp)+'_'+filt+data.arg.output+('_post' if post else '')+'.pyn'
    post_features = (0 if post else len(data.temp_ann_feature['names']))
    if os.path.exists(temp):
        with open(temp, mode='rb') as f:
            all_pos = pickle.load(f)
    else:
        all_pos = get_reduced_mat(method, data, sp, n_comp, filt, validation, ('annotation' not in sp and 'tissue' not in sp  and data.arg.each))
        with open(temp, mode='wb') as f:
            pickle.dump(all_pos, f)
        print('method', method, n_comp, n_comp_list)
    feat_names = [["PC" + '_' + str(i) for i in range(pos.shape[1])] for pos in all_pos]
    return all_pos, feat_names


def read_pvalue_based_features_from_file(fname):
    pvalue_filtered, min_pvalues = [], []
    print('# read from', fname)
    sys.stdout.flush()
    auc = False
    with open(fname) as f:
        for line in f.readlines():
            if len(line) == 0:
                continue
            contents = line.rstrip('\n').split('\t')
            if 'auc' in contents[0]:
                auc = True
            pvalue_filtered.append(int(contents[1]))
            if auc:
                min_pvalues.append(1.-float(contents[3]))
            else:
                min_pvalues.append(float(contents[3]))
    print('# end')
    sys.stdout.flush()
    return pvalue_filtered, min_pvalues, auc

def read_pvalue_based_features(ann, cindex, data, header, sp):
    global FEATURE, AUROC
    prefix = ('auc' if AUROC else 'pvalue')
    fname = prefix+'_list'+header+'_'+str(ann)+sp+'.txt'
    if os.path.exists(fname):
        pvalue_filtered, min_pvalues, auc = read_pvalue_based_features_from_file(fname)
    else:
        pvalue_filtered, min_pvalues, auc = data.fet_cindex(ann, cindex)
        for i in pvalue_filtered[0:16]:
            if AUROC:
                plot_roc(data.feature_data[:,cindex[i]], data.full_annotations[ann,:], header + '_' + FEATURE[ann] + '_' + prefix + '_' + str(i) + sp)
            plot_boxplot(data.feature_data[:,cindex[i]], data.full_annotations[ann,:], header + '_' + FEATURE[ann] + '_' + prefix + '_' + str(i) + sp)
        with open(fname, 'w') as f:
            for i in pvalue_filtered:
                p, c = min_pvalues[i], cindex[i]
                f.write('\t'.join([prefix, str(c), data.feature_names[c], str(data.feature_group[c]), str(p)])+'\n')
    return pvalue_filtered, min_pvalues

def scikit_survival(data):
    global SEED
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    cv_results = cross_validate(
        neigh, self.X, self.Y, scoring=self.metrics, cv=StratifiedKFold(n_splits=self.CV, shuffle=True, random_state=SEED))
    print('knn'+str(k), header, cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std(),
          cv_results['test_precision'].mean(), cv_results['test_recall'].mean())


def get_sp_range():
    return ['resnet', 'basic', 'pyradiomics', 'annotation', 'tissue', 'all']

def feature_based_classification(arg):
    global METHODS, FEATURE, AUROC, FVALIDATION
    np.random.seed(42)
    if arg.gbm:
        FEATURE = ['MGMT GBM', 'IDH GBM', 'TERT GBM']
    AUROC = arg.auc
    if len(arg.output) > 0:
        if arg.output[0] != '_':    arg.output = '_'+arg.output
    data = Dataset(arg)
    if arg.update:
        write_ann, write_feature, write_reduced_feature, write_validation = True, True, False, True
        print('Write validation?', write_validation, arg.pfile)
        if write_feature:           data.write_dataset(arg.output)
        if write_reduced_feature:   data.write_reduced_feature(arg.output)
        if write_validation and len(arg.pfile) > 1: data.write_valid_dataset(arg.output)
        if write_ann:               data.write_annotations(arg.output)
    data.load_dataset(arg.ann_feature, arg.filter, (not (arg.raw or arg.clust) and arg.post))
    if arg.clust:
        feature_and_ann_clustering(data, arg.ann_pred)
    if arg.filter:
        data.remove_incomplete_data()
    if arg.raw:
        for ann in range(len(FEATURE)):
            if ann != int(arg.sp):
                continue
            for sp in (['annotation'] if arg.ann_pred else ['all']):
                cindex = data.get_cindex_tag_based([sp], '')
                _, _ = read_pvalue_based_features(ann, cindex, data, arg.output, ('' if sp == 'all' else '_'+sp))
                all_pos = [data.feature_data[:,cindex]]
                if data.valid_data is not None:
                    all_pos.append(data.valid_data[:,cindex])
                all_pos = matrix_normalization(all_pos, (not arg.ann_pred and len(arg.ann_feature) > 0))
                for validation in FVALIDATION:
                    if data.valid_data is None and validation != 'test':
                        continue
                    header = "Raw_" + str(-1) + '_' + str(sp)+'_'+FEATURE[ann]+ '_' + validation
                    plot_prefix = arg.output+('_post' if arg.post else '')
                    if validation == 'both':
                        feature_filtering(data, all_pos[0], data.full_annotations[ann,:], ann,
                                            FEATURE[ann], header, vx=all_pos[1], vy=data.valid_annotations[ann,:], auc=arg.auc, plot_prefix=plot_prefix, gbm=arg.gbm)
                    elif validation == 'oppose':
                        feature_filtering(data, all_pos[1], data.valid_annotations[ann,:], ann,
                                            FEATURE[ann], header, vx=all_pos[0], vy=data.full_annotations[ann,:], auc=arg.auc, plot_prefix=plot_prefix, divide=True, gbm=arg.gbm)
                    elif validation == 'validation':
                        feature_filtering(data, all_pos[1], data.valid_annotations[ann,:], ann,
                                            FEATURE[ann], header, auc=arg.auc, plot_prefix=plot_prefix, gbm=arg.gbm)
                    else:
                        feature_filtering(data, all_pos[0], data.full_annotations[ann,:], ann,
                                            FEATURE[ann], header, auc=arg.auc, plot_prefix=plot_prefix, gbm=arg.gbm)

    if arg.mds:
        dimension_reduction('MDS', data, arg)
    if arg.pca:
        dimension_reduction('PCA', data, arg)
    if arg.spca:
        dimension_reduction('SPCA', data, arg)
    if arg.nmf:
        dimension_reduction('NMF', data, arg)
    if arg.snmf:
        dimension_reduction('SNMF', data, arg)
    return

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if len(args.ann_feature) == 0:
        args.post = False
    args = divide_validation_data(args)
    print(args)
    if args.time:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        feature_based_classification(args)
        pr.disable()
        pr.print_stats()
    else:
        feature_based_classification(args)
