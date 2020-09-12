#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: test_w2v_feature.py
@time: 2020/9/12 20:33
"""

import pandas as pd
import numpy as np
import os
import gc
import datetime as dt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models, similarities
from gensim.models.doc2vec import TaggedDocument

import gc
from base import Cache
from tqdm import tqdm


def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    tm_cols = df.select_dtypes('datetime').columns
    colsuse = [i for i in df.columns if i != 'label']
    for col in colsuse:
        if col in tm_cols:
            continue
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(
                        np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


if __name__ == "__main__":
    # base feature
    # data = Cache.reload_cache('CACHE_data_0912.pkl')
    # 造一份代码
    df = pd.DataFrame(np.random.randint(0, 21, (5000, 5)), columns=['uid', 'pt_d', 'task_id', 'adv_id', 'values'])
    df['task_id'] = df['task_id'] // 5
    df['adv_id'] = df['adv_id'] // 3
    df['pt_d'] = df['pt_d'] // 4
    df = df.sort_values(['uid', 'pt_d', 'task_id', 'adv_id']).reset_index(drop=True)
    df['label'] = np.random.randint(0, 2, (5000, 1))
    df = df.reset_index()
    df['communication_onlinerate'] = [' '.join(str(j) for j in np.random.randint(0, 25, (20,))) for i in
                                      range(df.shape[0])]
    # 过去一天的序列，做embedding
    # w2v

    # tfidf


