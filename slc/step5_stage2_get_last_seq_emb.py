#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gc
from base import Cache
from tqdm import tqdm
from gensim.models import Word2Vec
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('precision', 5)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)
# pd.set_option('max_colwidth', 200)
# pd.set_option('display.width', 5000)

print('start!')

def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
    tm_cols = df.select_dtypes('datetime').columns
    for col in df.columns:
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
    end_mem = df.memory_usage().sum() / 1024**2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_sequence(data,col,max_len=None):
    key2index = {}
    def split(x):
        for key in x:
            if key not in key2index:
                # Notice : input value 0 is a special "padding", 
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1 # 从1开始，0用于padding
        return list(map(lambda x: key2index[x], x))
    
    # preprocess the sequence feature
    id_list = list(map(split, data[col].values))# 转index
    id_list_length = np.array(list(map(len, id_list)))
    # max_len = max(genres_length)
    if max_len is None:
        max_len = int(np.percentile(id_list_length,99))
    id_list = pad_sequences(id_list, maxlen=max_len, padding='post',truncating='post')
    return id_list,key2index

def gen_list_df(feature):
    print(f'{feature} start!')
    data = Cache.reload_cache('CACHE_data_sampling_pos1_neg5.pkl')# 直接对采样后的数据做序列
    if feature =='label':
        data.loc[data['pt_d']>=8,'label'] = -1# test的label做mask
        data['label'] = data['label'].astype(np.int8)
        data['label'] = data['label']+ 1# 因为0用于padding置为0
    data = data[['uid',feature,'pt_d']]
    gc.collect()
    print(data.shape)
    data_group = data.groupby(['uid'])
    gc.collect()
    index_list = []
    feature_list = []
    print('index_list start')
    for name,group in tqdm(data_group):
        index_list.append(name)    
    print('feature_list start')
    for i in tqdm(index_list):
        index_get_group = data_group.get_group(i)
        ptd_set = set(index_get_group['pt_d'].values.flatten().tolist())
        for j in ptd_set:
            feature_list_ = []
            buf_list = []
            buf_list = index_get_group.query('pt_d < @j')[feature].values.flatten().tolist()# 本行样本之前的点击行为序列
            buf_list.append(0)# padding 0
            feature_list_.append(buf_list)# 行为序列
            feature_list_.append(j)# pt_d
            feature_list_.append(i)# uid
            feature_list.append(feature_list_)

    list_df = pd.DataFrame(feature_list)
    del index_list,feature_list,feature_list_,data_group,index_get_group,ptd_set
    gc.collect()
    list_df.columns=['list','pt_d','uid']
    list_df['list'] = list_df['list'].map(lambda x: [str(i) for i in x])# 转str
    list_df = list_df.drop_duplicates(subset=['pt_d','uid'])
    list_df = data.merge(list_df,how='left',on=('uid','pt_d'))# 顺序还是用data的顺序
    # 加入当天本样本 label不加
    if feature!='label':
        list_df['list'] = list_df[feature].map(lambda x:[str(x)]) + list_df['list']
    print('w2v start!')
    emb_size = 32# 预训练 embedding dim
    model = Word2Vec(
    list_df['list'].values.tolist(),
    size=emb_size,
    window=5,
    workers=5,
    min_count=1,
    sg=0, 
    hs=0, 
    negative=5,
    iter=5,
    seed=0)
    # 1 获取seq
    id_list,key2index = get_sequence(list_df,'list',max_len=40)
    # 2 获取key2index
    emb_dict = {}
    for word_i in list(model.wv.vocab.keys()):
        if word_i in model.wv:
            emb_dict[word_i] = model.wv[word_i]
        else:
            emb_dict[word_i] = np.zeros(emb_size)
    # 3 保存
    id_list_dict={}
    id_list_dict['id_list'] = id_list
    id_list_dict['key2index'] = key2index
    id_list_dict['emb'] = emb_dict
    Cache.cache_data(id_list_dict, nm_marker=f'EMB_INPUTSEQ_stage2_{feature}')
    print(f'{feature} done!')

from multiprocessing import Pool
if __name__ == '__main__':
    # 获取过去的list + 当前的一行
    # 得到id_list_dict和tx一样
    poc_feature_list = ['creat_type_cd','tags','spread_app_id','task_id','adv_id','label']#'task_id','adv_id','dev_id','inter_type_cd','spread_app_id','tags','app_first_class','app_second_class','his_app_size','his_on_shelf_time','app_score',,'creat_type_cd','adv_prim_id','indu_name'
    with Pool(6) as p:
        p.map(gen_list_df, poc_feature_list)
'''
以下是一些检查
'''
# #%%

# feature='label'
# print(f'{feature} start!')
# data = Cache.reload_cache('CACHE_data_step_1_feature_0917_r5.pkl')
# if feature =='label':
#     data['label'] = data['label'].fillna(2).astype(int)# mask到0
# #     data['label'] = data['label']+1# 因为0用于padding
# data = data[['uid',feature,'pt_d']]
# gc.collect()
# print(data.shape)
# data_group = data.groupby(['uid'])
# gc.collect()
# index_list = []
# feature_list = []
# print('index_list start')
# for name,group in tqdm(data_group):
#     index_list.append(name)    
# print('feature_list start')

# #%%

# feature_list=[]
# index_get_group = data_group.get_group(index_list[2000])
# ptd_set = set(index_get_group['pt_d'].values.flatten().tolist())
# for j in ptd_set:
#     feature_list_ = []
#     buf_list = []
#     buf_list = index_get_group.query('pt_d < @j')[feature].values.flatten().tolist()
#     buf_list.append(2)# padding 1
#     feature_list_.append(buf_list)# 行为序列
#     feature_list_.append(j)# pt_d
#     feature_list_.append(index_list[2000])# uid
#     feature_list.append(feature_list_)

# list_df = pd.DataFrame(feature_list)
# list_df

# #%%

# list_df.columns=['list','pt_d','uid']
# list_df['list'] = list_df['list'].map(lambda x: [str(i) for i in x])# 转str
# list_df = list_df.drop_duplicates(subset=['pt_d','uid'])
# #     data_uid_ptd = data[['uid','pt_d']]
# list_df = data.query('uid==1002170').merge(list_df,how='left',on=('uid','pt_d'))# 顺序还是用data的顺序
# # 加入当天本样本
# list_df['list'] = list_df[feature].map(lambda x:[str(x)]) + list_df['list']
# # list_df = list_df['list'].values.tolist()

# #%%

# list_df

# #%%

# print('w2v start!')
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# emb_size = 32
# model = Word2Vec(
# list_df['list'].values.tolist(),
# size=emb_size,
# window=5,
# workers=5,
# min_count=1,  # 最低词频. min_count>1会出现OOV
# sg=0,  # 1 for skip-gram; otherwise CBOW.
# hs=0,  # If 1, hierarchical softmax will be used for model training
# negative=5,  # hs=1 + negative 负采样
# iter=5,
# seed=0)
# # 1 获取seq
# id_list,key2index = get_sequence(list_df,'list',max_len=40)
# # 2 获取key2index
# emb_dict = {}
# for word_i in list(model.wv.vocab.keys()):
#     if word_i in model.wv:
#         emb_dict[word_i] = model.wv[word_i]
#     else:
#         emb_dict[word_i] = np.zeros(emb_size)
# # 3 保存
# id_list_dict={}
# id_list_dict['id_list'] = id_list
# id_list_dict['key2index'] = key2index
# id_list_dict['emb'] = emb_dict

# #%%

# id_list_dict

# #%%

# import pandas as pd
# import numpy as np
# import gc
# from base import Cache
# from tqdm import tqdm
# from gensim.models import Word2Vec
# data = Cache.reload_cache('CACHE_data_step_1_feature_0917_r5.pkl')
# seq_emb = Cache.reload_cache('CACHE_EMB_INPUTSEQ_adv_id.pkl')

# #%%

# data[['index','uid','pt_d','adv_id']].head()

# #%%

# data.query('uid==2237673')[['index','uid','pt_d','adv_id']]

# #%%

# seq_emb.keys()

# #%%

# seq_emb['id_list'][:5,:]

# #%%

# seq_emb['key2index']['6340']

# #%%

# seq_emb['key2index']['4501']

# #%% md

# ### 检查过了，seq 和 dense 可以对上！

# #%%

