import sys
import numpy as np
import os

def parse_file_list(name, std_brn, single=True):
    if single:
        dirs = name.split(',')[0].split('/')
        patient, name = dirs[-2], dirs[-1]
        for m in ['GD', 'GT', 'FLAIR', 'T1', 'T2']:
            if m in name:
                return patient + '__' + std_brn + '_' + m + '_' + name[0], patient + '__' + std_brn + '_' + m
    else:
        return None

def convert_to_vector(features):
    me = np.mean(np.array(features), axis=0)
    su = np.sum(np.array(features), axis=0)
    print(me.shape, su.shape)
    return(np.append(me, su))

def print_data(sample, features, file, append=False):
    vec = convert_to_vector(features)
    print(len(vec), file, sample)
    if append:
        flag = 'a'
    else:
        flag = 'w'
    with open(file, flag) as f:
        if not append:
            f.write('\t'.join(['sample']+[str(i)+'_'+sp for sp in ['m', 's'] for i in range(int(len(vec)/2))])+'\n')
        f.write(sample)
        for v in vec:
            f.write('\t'+str(v))
        f.write('\n')


def main(file, prefix, std_brn='', append=False):
    global BRAIN
    BRAIN = std_brn
    if not os.path.exists(file):
        return
    with open(file) as f:
        pre, pfprefix = None, None
        features = []
        for line in f.readlines():
            contents = line.rstrip('\n').split('\t')
            pname, fprefix = parse_file_list(contents[0], std_brn, single=True)
            if pname == pre:
                features.append(list(map(float, contents[2:])))
            else:
                if pre != None:
                    print_data(pre, features, prefix + pfprefix + '.tsv', append)
                pre, pfprefix = pname, fprefix
                features = []
        if pre != None:
            print_data(pre, features, prefix + pfprefix + '.tsv', append)



if __name__ == '__main__':
    if len(sys.argv) > 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        header = 'r'
        for norm in ['k', 'z']:
            oheader = norm
            for type in ['gbm', 'gbm_edema', 'glioma']:
                if len(header) > 0:
                    main(header+norm+'2wfiles_'+type+'_cpu.res', oheader+'res_'+type+'_', '2w')
                    main(header+norm+'afiles_'+type+'_cpu.res', oheader+'res_'+type+'_', 'a')
                    main(header+norm+'files_'+type+'_cpu.res', oheader+'res_'+type+'_', '')
                    main(header+norm+'sfiles_'+type+'_cpu.res', oheader+'bres_'+type+'_', 's')
                    main(header+norm+'eafiles_'+type+'_cpu.res', oheader+'bres_'+type+'_', 'ea')
                    main(header+norm+'s2wfiles_'+type+'_cpu.res', oheader+'bres_'+type+'_', 's2w')
                if True:
                    main(norm+'files_'+type+'_cpu.res', oheader+'ares_'+type+'_', '')
                    main(norm+'afiles_'+type+'_cpu.res', oheader+'ares_'+type+'_', 'a')
                    main(norm+'2wfiles_'+type+'_cpu.res', oheader+'ares_'+type+'_', '2w')
                else:
                    main(norm+'sfiles_'+type+'_cpu.res', oheader+'ares_'+type+'_', 's')
                    main(norm+'eafiles_'+type+'_cpu.res', oheader+'ares_'+type+'_', 'ea')
                    main(norm+'s2wfiles_'+type+'_cpu.res', oheader+'ares_'+type+'_', 's2w')
