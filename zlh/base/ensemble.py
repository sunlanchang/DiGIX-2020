#!/usr/bin/env python
# encoding: utf-8

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: ensemble.py
@time: 2020/2/22 16:58
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

    def auto_ensemble(self, data_list, dfLabel, collist, max_iter, min_use=3, seed=0, metric='ks', weight_oot=0.9,
                      thre=0.9):
        random.seed(seed)
        listu = []
        for i in range(min_use, len(collist) + 1):
            iter = itertools.permutations(collist, i)
            for i in list(iter):
                listu.append(i)
        random.shuffle(listu)
        bestmetric = 0
        bestlist = []
        iteri = 0
        for listi in tqdm(listu):
            df = self.get_finial_subfile(data_list=data_list, listmodel=list(listi), printflag=False, path=False,
                                         thre=thre)
            # 查看融合结果ks
            df = df.merge(dfLabel[['id', 'label', 'app_time', 'null_nums']], on='id', how='left')
            dfLcheck_oot = df.loc[(df['app_time'] >= '2019-11-01') & (df['app_time'] < '2020-04-01')].copy()
            dfLcheck_oot = dfLcheck_oot.query('null_nums == 0 and label>=0')  # 只看非空的
            dfLcheck_trnval = df.loc[(df['app_time'] < '2019-11-01')]
            dfLcheck_trnval = dfLcheck_trnval.query('label>=0')
            if metric == 'ks':
                metric_oot = getks(dfLcheck_oot, 'prob')
                metric_trnval = getks(dfLcheck_trnval, 'prob')
            else:
                metric_oot = getauc(dfLcheck_oot, 'prob')
                metric_trnval = getauc(dfLcheck_trnval, 'prob')
            nowmetric = metric_oot * weight_oot + metric_trnval * (1 - weight_oot)
            if nowmetric > bestmetric:
                bestmetric = nowmetric
                bestlist = list(listi)
                print('find best {} :{} order:{}'.format(metric, bestmetric, bestlist))
            iteri += 1
            if iteri > max_iter:
                break
        return bestlist

    def get_finial_subfile(self, data_list, path='./ensemble_files/', listmodel=[], printflag=False, thre=0.97,
                           output_path=''):
        if path != False:
            data_list = []
            flag = 0
            for info in os.listdir(path):
                domain = os.path.abspath(path)
                info = os.path.join(domain, info)
                data = pd.read_csv(info)
                data.sort_values('id', ascending=True, inplace=True)
                data.reset_index(drop=True, inplace=True)
                data['flag'] = flag
                flag += 1
                # print(info, data.iloc[:, 1].mean(), data.iloc[:, 1].max(), data.iloc[:, 1].min())
                data_list.append(data)
        ensemblelist = []
        corlist = []
        for dfi in data_list:
            corlist.append(dfi.iloc[:, 1])
        cor_df = pd.concat(corlist, axis=1)
        cor_df = cor_df[listmodel]  # 按此顺序
        corr = cor_df.corr()# (method="spearman")
        print(corr)
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
            # print(indexn, mx, w)
            m_gmean += w * (np.log(cor_df.iloc[:, mx[0]]) + np.log(cor_df.iloc[:, mx[1]])) / 2
            if printflag:
                print('(np.log(', listmodel[mx[0]] + ')+' + 'np.log(', listmodel[mx[1]] + '))/2*', str(w), '''+''')
            s += w
            rank[mx] = 1
        if printflag:
            print('finial')
        m_gmean = np.exp(m_gmean / s)
        if printflag:
            print('np.exp(' + 'm_gmean' + '/{})'.format(s))
        ensemblelist.append(pd.DataFrame(m_gmean))
        df_ensemble = pd.concat(ensemblelist, axis=1)
        df_ensemble.columns = ['prob']
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
    pe.get_finial_subfile([])
