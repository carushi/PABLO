import argparse
import numpy as np
np.set_printoptions(threshold=np.inf)
import pickle
SEED = np.random.RandomState(seed=3)


SAMPLE = ['gbm', 'glioma']
METHODS = {'gbm': ['GD', 'T1', 'T2', 'FLAIR'], 'glioma': ['GT', 'T1', 'T2', 'FLAIR'],
           'tcia_gbm': ['GD', 'T1', 'T2', 'FLAIR'], 'tcia_glioma':['GD', 'T1', 'T2', 'FLAIR']}
UNIQ_METHODS = list(set([elem for dict in METHODS for elem in dict]))

# Set the annotation column names
FEATURE = ['GBM', 'CpG', 'IDH1_2', 'TERT', '1p19q', 'MGMT', 'Group A', 'Group B', 'Group C', 'Group D', 'MGMT GBM', 'MGMT LGG', 'IDH GBM', 'TERT GBM', 'TERT LGG', 'IDH LGG']


MAX = 1000 # Max patient id
CV = 10 # Cross validation
AUROC = True # pvalue or auroc
SPMTEMP = "../ann/TPM.nii"
ANTSTEMP = "../ann/priors[NUM].nii.gz"
SCALE = True


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regex', dest='reg_file', type=str, default='prefix_regex.tsv')
    parser.add_argument('--normalize_regexp', dest='normalize', type=str,
                        help='', default='[ID]__[NORM]_[METHOD]_k')
    parser.add_argument('--output', dest='output', type=str, default='')
    parser.add_argument('--update', dest='update', action='store_true', help='update data matrix (*.pyn)')
    parser.add_argument('--dir', dest='dir', type=str, default='./')
    parser.add_argument('--dim_list', dest='dim_list', type=str, default='')
    parser.add_argument('--ann', dest='ann', type=str)
    parser.add_argument('--ann_pred', dest='ann_pred', action='store_true')
    parser.add_argument('--ann_feature', dest='ann_feature', type=str, default="")
    parser.add_argument('--post', dest='post', action='store_true', help="merge annotation feature after compression")
    parser.add_argument('--patient', dest='pfile', type=str, default='patient_id_list.tsv')
    parser.add_argument('--raw', dest='raw', action='store_true')
    parser.add_argument('--clust', dest='clust', action='store_true')
    parser.add_argument('--fisher', dest='fisher', action='store_true')
    parser.add_argument('--mds', dest='mds', action='store_true')
    parser.add_argument('--pca', dest='pca', action='store_true')
    parser.add_argument('--spca', dest='spca', action='store_true')
    parser.add_argument('--nmf', dest='nmf', action='store_true')
    parser.add_argument('--snmf', dest='snmf', action='store_true')
    parser.add_argument('--sp', dest='sp', type=str, default=None)
    parser.add_argument('--auc', dest='auc', action='store_true')
    parser.add_argument('--time', dest='time', action='store_true')
    parser.add_argument('--gbm', dest='gbm', action='store_true')
    parser.add_argument('--each', dest='each', action='store_true')

    parser.add_argument('--res_regexp', dest='res', type=str, default='res_[GRADE]_[ID]___[METHOD].tsv')
    parser.add_argument('--feature_prefix', dest='feature', type=str,
                        help='foo help', default='py_[GRADE]_[ID]___[METHOD].tsv,b[GRADE]_[ID]___[METHOD].tsv')
    parser.add_argument('--tumor_map', dest='tumor_map', type=str, default='bs[GRADE]_[ID]_2w_[METHOD].tsv')
    parser.add_argument('--filter', dest='filter', action='store_true')
    parser.add_argument('--life', dest='life', action='store_true')
    return parser

def divide_validation_data(parser):
    parser.reg_file = parser.reg_file.split(',')
    parser.dir = parser.dir.split(',')
    parser.normalize = parser.normalize.split(',')
    parser.pfile = parser.pfile.split(',')
    assert all(x == len(parser.reg_file) for x in [len(parser.reg_file), len(parser.dir), len(parser.normalize), len(parser.pfile)])
    return parser

def normalize_rows(M):
    row_sums = M.sum(axis=1)
    return M / row_sums

def whitend_mat(M):
    return whitening(M)

def remove_nan(M, max_val=1e10):
    M[np.isinf(M)] = float('nan')
    M = np.nan_to_num(M)
    M[np.logical_not(np.isfinite(M))] = 0.
    M[np.where(M > max_val)] = max_val
    M[np.where(M < -max_val)] = -max_val
    return M

def normalize_mat(M, axis=1):
    M[np.isinf(M)] = float('nan')
    M = np.nan_to_num(M)
    col = M.max(axis=axis)
    col = np.clip(col, 1., max(col))
    return (M.transpose()/col).transpose()

def normalize_row_col_mat(M):
    M[np.isinf(M)] = float('nan')
    M = np.nan_to_num(M)
    for a in [0, 1]:
        if a == 0:
            M = M/np.maximum(np.ones(shape=M.shape), M.max(axis=a))
        # else:
        #     mmean = M.mean(axis=1)
        #     mstd = M.std(axis=1)
        #     M = ((M.transpose() - mmean) / mstd).transpose()
    return M


def replace_patient_method(seq, patient, method, grade, norm=""):
    return seq.replace('[ID]', patient).replace('[METHOD]', method).replace('[GRADE]', grade).replace('[NORM]', norm)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        return super().find_class(module, name)

class FeatureLibrary:
    def __init__(self, dicts=None):
        if dicts is None:
            self.dicts = []
        else:
            self.dicts = dicts
    def insert(self, dict):
        if dict in self.dicts:
            return self.dicts.index(dict)
        else:
            self.dicts.append(dict)
            return len(self.dicts)-1
    def is_valid(self, condition):
        if isinstance(condition, (list,)):
            return [i for i in range(len(self.dicts)) if len([c for c in condition if c in self.dicts[i]]) == len(condition)]
        else:
            return [i for i in range(len(self.dicts)) if condition in self.dicts[i]]
