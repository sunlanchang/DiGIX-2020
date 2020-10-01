#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: ensemble.py
@time: 2020/09/29 20:58
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from tqdm import tqdm
import os
import itertools
import random

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = 1000
from sklearn.metrics import roc_curve, roc_auc_score


def getks(x, var):
    fpr_lr_train_valid, tpr_lr_train_valid, _ = roc_curve(x['label'], x[var])
    vaild_ks = abs(fpr_lr_train_valid - tpr_lr_train_valid).max()
    return vaild_ks


def getauc(x, var):
    auci = roc_auc_score(x['label'], x[var])
    return auci


class process_emsemble:
    def __init__(self):
        pass

    def get_finial_subfile(self, data_list, path='./ensemble_files/', listmodel=[], printflag=False, thre=0.97,
                           output_path=''):
        if path != False:
            data_list = []
            for info in sorted(os.listdir(path))[::-1]:
                domain = os.path.abspath(path)
                info_ = os.path.join(domain, info)
                data = pd.read_csv(info_)
                data.sort_values('id', ascending=True, inplace=True)
                info = info.split('.')[0]
                data.columns = ['id', f'prob_{info}']
                data.reset_index(drop=True, inplace=True)
                print(info, data.iloc[:, 1].mean(), data.iloc[:, 1].max(), data.iloc[:, 1].min())
                if data.iloc[:, 1].max() < 0.1:
                    data.iloc[:, 1] = data.iloc[:, 1] * 10
                data_list.append(data)
                listmodel.append(data.columns[-1])
        ensemblelist = []
        corlist = []
        for dfi in data_list:
            corlist.append(dfi.iloc[:, 1])
        cor_df = pd.concat(corlist, axis=1)
        cor_df = cor_df[listmodel]  # 按此顺序
        print(listmodel)
        corr = cor_df.corr('spearman')  # 'spearman'
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        rank = np.tril(corr.values, -1)
        m = (rank > 0).sum() - (rank > thre).sum()
        if printflag:
            print('total:', m)
        m_gmean, s = 0, 0
        for indexn, n in enumerate(range(m)):
            mx = np.unravel_index(rank.argmin(), rank.shape)
            w = (m - n) / m
            if printflag:
                print(indexn, mx, w)
            m_gmean += w * (np.log(cor_df.iloc[:, mx[0]]) + np.log(cor_df.iloc[:, mx[1]])) / 2
            if printflag:
                print('(np.log(', listmodel[mx[0]] + ')+' + 'np.log(', listmodel[mx[1]] + '))/2*', str(w), '''+''')
            s += w
            rank[mx] = 1
        if printflag:
            print('finial')
        if s == 0:
            print([dfi.columns for dfi in data_list], corr)
        m_gmean = np.exp(m_gmean / s)
        if printflag:
            print('np.exp(' + 'm_gmean' + '/{})'.format(s))
        ensemblelist.append(pd.DataFrame(m_gmean))
        df_ensemble = pd.concat(ensemblelist, axis=1)
        df_ensemble.columns = ['probability']
        df_ensemble['id'] = data_list[0]['id']
        modellist = '_'.join(listmodel)
        if printflag:
            df_ensemble.to_csv(
                output_path + 'ensemble_file_{}.csv'.format(modellist),
                index=False,
                encoding='utf-8')
            print('finial subfile: {} done!'.format(modellist))
        return df_ensemble


if __name__ == "__main__":
    pe = process_emsemble()
    df_ensemble = pe.get_finial_subfile(data_list=[], path='./sub_use/', printflag=True)
    df_ensemble = df_ensemble[['id','probability']]
    df_ensemble['id'] = df_ensemble['id'].astype(int)
    df_ensemble.to_csv('submission_ensemble_1.csv', index=False)# 1

    path = './sub_use/'
    df1 = pd.read_csv(path + 'submission_08004.csv')
    df2 = pd.read_csv(path + 'submission_08005.csv')
    df1 = df1.merge(df2, on='id', how='left')
    df1['probability'] = np.exp((np.log(df1['probability_x']) + np.log(df1['probability_y'] * 10)) / 2)
    df1[['id', 'probability']].to_csv('submission_two_low.csv', index=False)


    # finial
    df1 = pd.read_csv('a-fusion-0.807027.csv')# 两个最高分的均值 base 由队友在组队前初次提交应该 （submission_08033+submission_08032）/2
    df2 = pd.read_csv('submission_ensemble_1.csv')# gmeans all
    df3 = pd.read_csv('submission_two_low_0804613.csv')# 两个最低分的log均值
    df_ensemble = df1.copy()
    df_ensemble['probability'] = df_ensemble['probability'] * 0.5 + df2['probability'] * 0.31 + df3[
        'probability'] * 0.19
    df_ensemble.to_csv('submission_ensemble_finial.csv', index=False)