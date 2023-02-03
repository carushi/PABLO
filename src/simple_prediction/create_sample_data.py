#!/usr/bin/env python3  Line 1
# -*- coding: utf-8 -*-

import random
import pandas as pd
import numpy as np

col_label = ['ID', 'Dataset', 'Diag','16CpG','IDH1/2','Group','TERT','1p19q','Age_n','Sex','KPS_n']
df = []
for db in ['NCC', 'TCIA']:
    for i in range(1, 101):
        vec = [str(int(i)), db]
        for f in col_label[2:len(col_label)]:
            if f in ['16CpG', 'Age_n']:
                max_value = {'16CpG':20, 'Age_n':80}[f]
                value = random.uniform(0.0, max_value)
                if f == 'Age_n':
                    vec.append(int(value))
                else:
                    vec.append(value)
            else:
                select_value = {'Diag':('gbm', 'glioma'), 'IDH1/2':('Yes', 'wt'), 
                                'Group':('A', 'B', 'C', 'D'), 'TERT':('Yes', 'wt'), '1p19q':('co-del', 'no loss'), 'Sex':('F', 'M'), 'KPS_n':('50', '60', '70', '80', '90', '100') }
                vec.append(random.choice(select_value[f]))
        df.append(vec)

pd.DataFrame(df, columns=col_label).to_csv('sample_annotation.csv', index=None)


col_label = ['UniqueID']+['feature'+str(i) for i in range(1, 1001)]
A = np.random.normal(0, 1, (200, 1001))
print(np.concatenate((np.linspace(1, 100, 100), np.linspace(1, 100, 100))).shape)
A[:,0] = np.concatenate((np.linspace(1, 100, 100), np.linspace(1, 100, 100)))
pd.DataFrame(A, columns=col_label).to_csv('sample_feature.csv', index=None)


