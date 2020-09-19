#!/usr/bin/env python 
# encoding: utf-8 
## w2v 特征 不用跑完，抽几个跑
"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: data_reprocess_step3.py
@time: 2020/9/13 21:31
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
import multiprocessing


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


'''
# 封装类
class text_emb_tool:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.temp_list = []
        self.temp_index = []
        self.dictionary = []
        self.corpus = []
        self.corpus_tfidf = []

    def func_pad(self, arr):
        arr = arr.T
        arr = pad_sequences(arr, maxlen=self.seq_len, padding='pre',
                            truncating='pre', dtype='float32', value=0.0)
        arr = arr.T
        return arr

    def get_tmp_list(self, col):
        arr = id_list_dict[col+'_list']['id_list'].copy()
        arr = arr.astype(str)
        self.temp_list = []
        for i in tqdm(range(len(arr))):
            tpi = list(filter(lambda number: number != '0', arr[i]))
            self.temp_list.append(tpi)
        # get tfidf embedding
        print('dictionary start!')
        self.dictionary = corpora.Dictionary(self.temp_list, prune_at=None)
        self.corpus = [self.dictionary.doc2bow(
            text) for text in self.temp_list]
        self.corpus_tfidf = models.TfidfModel(self.corpus)[self.corpus]
        print('dictionary finish!')
        # 序列转词表索引
        print('dictionary index start!')
        self.temp_index = []
        for si in tqdm(self.temp_list):
            sentencei = [self.dictionary.token2id[i] for i in si]
            self.temp_index.append(sentencei)
        print('dictionary index finish!')

    def get_embeddings(self):
        tfidf_dict = {}
        array_tfidf = []
        for index, sentencei in enumerate(self.temp_list):
            if index % 100000 == 0:
                print(index)
            tfidf_dict = dict(self.corpus_tfidf[index])  # 词表索引：tfidf
            # sentencei-># 词表索引
            sentencei_index = self.temp_index[index]
            # tfidf
            tfidfs = list(map(lambda x: tfidf_dict[x], sentencei_index))
            array_tfidf.append(tfidfs)
        tfidfs = pad_sequences(array_tfidf, maxlen=self.seq_len,
                               padding='pre', truncating='pre', dtype='float32', value=0.0)
        array_tfidf = np.array(tfidfs)  # id * seqlen
        array_tfidf = array_tfidf.reshape(
            len(self.temp_list), -1, 1)  # id seqlen 1
        print('return tfidf matrix shape:', array_tfidf.shape)
        return array_tfidf

#%%

# 每个列提取哪些emb
cols_to_emb = ['creative_id', 'ad_id', 'advertiser_id', 'product_id','product_category', 'industry','time']
# 开跑
import gc
# id_list_dict = {}
# for key,value in id_list_dict0.items():
#     id_list_dict[key]={}
#     id_list_dict[key]['id_list']=value['id_list'][:500,:]
for var in cols_to_emb:
    res_dict = {}
    tet = text_emb_tool(seq_len = 120)
    tet.get_tmp_list(var)# 字典 索引
    emb_matrix = tet.get_embeddings()
    res_dict['word_emb_dict'] = emb_matrix
    print(res_dict)
    Cache.cache_data(res_dict,nm_marker=f'EMB_DICT_TEXTEMB_{var}_tfidf')
    print(f'EMB_DICT_TEXTEMB_{var}'+' finished!')
    del tet
    gc.collect()
print('text_embedding_finished!')
'''


def get_embedding_pro(df_raw, sentence_id, word_id, emb_size=128, window=10,
                      dropna=False, n_jobs=4, method='skipgram',
                      hs=0, negative=10, epoch=10, return_model=False,
                      embedding_type='fasttext', slide_window=1):
    """
    Now, set min_count=1 to avoid OOV...
    How to deal with oov in a more appropriate way...
    Paramter:
    ----------
    df_raw: DataFrame contains columns named sentence_id and word_id
    sentence_id: like user ID, will be coerced into str
    word_id: like item ID, will be coerced into str
    emb_size: default 8
    dropna: default False, nans will be filled with 'NULL_zhangqibot'. if True, nans will all be dropped.
    n_jobs: 4 cpus to use as default
    method: 'sg'/'skipgram' or 'cbow'
        sg : {0, 1}, optional
            Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs : {0, 1}, optional
        If 1, hierarchical softmax will be used for model training.
        If 0, and `negative` is non-zero, negative sampling will be used.
    negative : int, optional
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    epoch: iter : int, optional,default 10
        Number of iterations (epochs) over the corpus.
    return_model: default True
    embedding_type: fasttext word2vec
    Return:
    ----------
    Example:
    def run_w2v(sentence_id,word_id,emb_size=128):
        res_dict= w2v_pro(datalog,sentence_id=sentence_id,word_id=word_id,
                          emb_size=emb_size,dropna=False,n_jobs=-1,
                          method='cbow', hs=0,negative=10,epoch=10,
                          return_model=False)
        Cache.cache_data(res_dict,nm_marker=f'EMB_DICT_W2V_CBOW_10EPOCH_{sentence_id}_{word_id}')

    sentence_id='user_id'
    for word_id in tqdm(['creative_id', 'ad_id', 'product_id', 'advertiser_id']):
        run_w2v(sentence_id,word_id,emb_size=128)

    run_w2v(sentence_id,word_id='product_category',emb_size=8)
    run_w2v(sentence_id,word_id='industry',emb_size=64)
    ----------
    """
    if method.lower() in ['sg', 'skipgram']:
        sg = 1
    elif method.lower() in ['cbow']:
        sg = 0
    else:
        raise NotImplementedError
    list_col_nm = f'{sentence_id}__{word_id}_list'
    if (n_jobs is None) or (n_jobs <= 0):
        n_jobs = multiprocessing.cpu_count()
    print(f"========== W2V:  {sentence_id} {word_id} ==========")

    df = df_raw[[sentence_id, word_id, 'pt_d']].copy()

    if df[sentence_id].isnull().sum() > 0:
        print("NaNs exist in sentence_id column!!")
    if dropna:
        df = df.dropna(subset=[sentence_id, word_id])
    else:
        df[word_id] = df[word_id].fillna(-1).astype(int).astype(str)
        df[sentence_id] = df[sentence_id].fillna(-1).astype(int).astype(str)

    df['pt_d_last'] = df['pt_d'] + slide_window
    fe = df.groupby([sentence_id, 'pt_d_last'])[word_id].apply(lambda x: list(x)).reset_index()
    fe.columns = [sentence_id, 'pt_d', list_col_nm]
    df = df.merge(fe, on=[sentence_id, 'pt_d'], how='left')
    df[list_col_nm] = df[list_col_nm].map(lambda x: x if isinstance(x, list) else [])
    # 加上本行的
    df[word_id + '_add'] = df[word_id].map(lambda x: [x])
    df[list_col_nm] = df[list_col_nm] + df[word_id + '_add']
    sentences = df[list_col_nm].values.tolist()
    all_words_vocabulary = df[word_id].unique().tolist()
    del df[list_col_nm], df['pt_d_last'], df[word_id + '_add']
    gc.collect()
    if embedding_type == 'w2v':
        model = Word2Vec(
            sentences,
            size=emb_size,
            window=window,
            workers=n_jobs,
            min_count=1,  # 最低词频. min_count>1会出现OOV
            sg=sg,  # 1 for skip-gram; otherwise CBOW.
            hs=hs,  # If 1, hierarchical softmax will be used for model training
            negative=negative,  # hs=1 + negative 负采样
            iter=epoch,
            seed=0)
    else:
        model = models.FastText(sentences, size=emb_size,
                                window=window, workers=n_jobs, seed=0, sg=sg, iter=epoch)

    # get word embedding matrix
    emb_dict = {}
    for word_i in all_words_vocabulary:
        if word_i in model.wv:
            emb_dict[word_i] = model.wv[word_i]
        else:
            emb_dict[word_i] = np.zeros(emb_size)

    # get sentence embedding matrix
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    emb_cols = []
    for i in range(emb_size):
        df[f'EMB_{embedding_type}_{sentence_id}_{word_id}_{slide_window}_emb_{i}'] = emb_matrix[:, i]
        emb_cols.append(f'EMB_{embedding_type}_{sentence_id}_{word_id}_{slide_window}_emb_{i}')

    if not return_model:
        model = None
    return {"word_emb_dict": emb_dict, "sentence_emb_df": df[emb_cols], 'model': model}


if __name__ == "__main__":
    # base feature
    df = Cache.reload_cache('CACHE_data_0912.pkl')
    df['communication_onlinerate'] = df['communication_onlinerate'].map(lambda x: x.split(' '))
    df['communication_onlinerate'] = df['communication_onlinerate'].map(lambda x: x if isinstance(x, list) else [])
    n_jobs = 40
    sparse_features = ['creat_type_cd', 'slot_id',
                       'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career',
                       'gender', 'net_type', 'residence', 'emui_dev', 'indu_name', 'age', 'label']
    '''
    ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 
                     'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career',
                     'gender', 'net_type', 'residence', 'emui_dev', 'indu_name', 'age', 'city_rank','label']
    '''
    dense_features = ['his_app_size', 'his_on_shelf_time', 'app_score', 'device_size', 'list_time', 'device_price',
                      'communication_avgonline_30d', 'cmr_None', 'sample_nunique']
    df = df[['index', 'uid', 'pt_d', 'communication_onlinerate'] + sparse_features + dense_features]
    gc.collect()
    #     # 造一份代码
    #     df = pd.DataFrame(np.random.randint(0, 21, (5000, 5)), columns=['uid', 'pt_d', 'task_id', 'adv_id', 'values'])
    #     df['task_id'] = df['task_id']
    #     df['adv_id'] = df['adv_id']
    #     df['pt_d'] = df['pt_d'] // 5
    #     df = df.sort_values(['uid', 'pt_d', 'task_id', 'adv_id']).reset_index(drop=True)
    #     df['label'] = np.random.randint(0, 2, (5000, 1))
    #     df = df.reset_index()
    #     df['communication_onlinerate'] = [' '.join(str(j) for j in np.random.randint(0, 25, (20,))) for i in
    #                                       range(df.shape[0])]
    # 直接做embedding
    sentences = df['communication_onlinerate'].values.tolist()
    model = Word2Vec(
        sentences,
        size=8,
        window=5,
        workers=4,
        min_count=1,  # 最低词频. min_count>1会出现OOV
        sg=0,  # 1 for skip-gram; otherwise CBOW.
        hs=0,  # If 1, hierarchical softmax will be used for model training
        negative=5,  # hs=1 + negative 负采样
        iter=5,
        seed=0)
    # get sentence embedding matrix
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model.wv:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * 8)
    emb_matrix = np.array(emb_matrix)
    add_cols = []
    for i in range(8):
        df[f'EMB_w2v_uid_communication_onlinerate_0_emb_{i}'] = emb_matrix[:, i]
        add_cols.append(f'EMB_w2v_uid_communication_onlinerate_0_emb_{i}')
    Cache.cache_data(df[['index'] + add_cols], nm_marker=f'EMB_DICT_8_1_8_uid_communication_onlinerate_w2v')
    del df['communication_onlinerate'], model, emb_matrix, sentences
    df = df.drop(add_cols, axis=1)
    gc.collect()

    # 过去一天的序列，做embedding
    df = reduce_mem(df, use_float16=True)
    gc.collect()


    # w2v
    def run_w2v(df, sentence_id, word_id, emb_size=256, window=10, slid_window=1, embedding_type='w2v', n_jobs=72):
        res_dict = get_embedding_pro(df, sentence_id=sentence_id, word_id=word_id, window=window,
                                     slide_window=slid_window,
                                     emb_size=emb_size, dropna=False, n_jobs=n_jobs, return_model=False, epoch=5,
                                     embedding_type=embedding_type)

        return res_dict["sentence_emb_df"]


    emb_size_dict = {}
    for var in tqdm(sparse_features + dense_features):
        nunique_nums = df[var].nunique()
        emb_dim = max([nunique_nums // 50, 8])
        emb_dim = 32 if emb_dim > 16 else (16 if emb_dim > 8 else 8)
        emb_size_dict[var] = emb_dim
        print(var, ' emb_dim: ', emb_dim)
        fe = run_w2v(df, 'uid', var, emb_size=emb_dim, window=8, slid_window=1, embedding_type='w2v', n_jobs=n_jobs)
        fe['index'] = df['index']
        Cache.cache_data(fe, nm_marker=f'EMB_DICT_8_1_{emb_dim}_uid_{var}_w2v')# 有index
#         fe = run_w2v(df, 'uid', var, emb_size=emb_dim, window=8,slid_window=1, embedding_type='w2v', n_jobs = n_jobs)
#         fe = reduce_mem(fe, use_float16=True)
#         df = pd.concat([df, fe], axis=1)
#         fe = run_w2v(df, 'uid', var, emb_size=emb_dim, window=8,slid_window=1, embedding_type='fasttext', n_jobs = n_jobs)
#         fe = reduce_mem(fe, use_float16=True)
#         df = pd.concat([df, fe], axis=1)

#         fe = run_w2v(df, 'uid', var, emb_size=emb_dim, window=16,slid_window=2, embedding_type='w2v', n_jobs = n_jobs)
#         fe = reduce_mem(fe, use_float16=True)
#         df = pd.concat([df, fe], axis=1)
#         fe = run_w2v(df, 'uid', var, emb_size=emb_dim, window=16,slid_window=2, embedding_type='fasttext', n_jobs = n_jobs)
#         fe = reduce_mem(fe, use_float16=True)
#         del df[var]
#         gc.collect()
#         df = pd.concat([df, fe], axis=1)

#     cols_to_save = [i for i in df.columns if i.find('EMB_') > -1]
#     df = df[['index'] + cols_to_save]
#     Cache.cache_data(df, nm_marker='EMB_feature0912')

