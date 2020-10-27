import pandas as pd
import numpy as np
import gc
from base import Cache
from tqdm import tqdm
from gensim.models import Word2Vec
import sys
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)


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


def gen_list_df(data, item_fe_list):
    setitem_fe_list = data.groupby(item_fe_list)[
        'raw_index'].count().reset_index()  # 所有feature1
    del setitem_fe_list['raw_index']
    setkey = set(data['user_f'].astype(str))
    ## 所有点击行为的
    list_df = data.query('label==1').groupby(item_fe_list)['user_f'].apply(lambda x: list(x)).\
        rename('list_user_f').reset_index()
    list_df = setitem_fe_list.merge(list_df, on=item_fe_list, how='left')
    list_df['list_user_f'] = list_df['list_user_f'].map(
        lambda x: x if isinstance(x, list) else ['N'])
    list_df_list = list_df['list_user_f'].values.tolist()
    print('w2v start!')
    model = Word2Vec(
        list_df_list,
        size=128,
        window=10000,  # 增加了10倍
        workers=10,
        min_count=1,
        sg=0,
        hs=0,
        negative=5,
        iter=10,
        seed=0)
    emb_matrix = []
    for seq in list_df_list:  # 对应每个'item_f'
        vec = []
        for w in seq:
            vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * 128)

    emb_matrix = np.array(emb_matrix, dtype=np.float32)
    dict_feature1 = {}
    for index, key in enumerate(list(list_df[item_fe_list])):
        dict_feature1[key] = emb_matrix[index, :]  # feature1的字典 key是item_f
    # 直接存emb
    # index,emb_item,emb_user
    np.save('./cached_data/dataindex_stage2.npy', data['raw_index'].values)
    m1_item = []
    for i in tqdm(list(data[item_fe_list])):
        m1_item.append(dict_feature1[i])
    m1_item = np.array(m1_item, dtype=np.float32)
    m1_user = []
    for i in tqdm(list(data['user_f'])):
        try:
            m1_user.append(model.wv[i])
        except:
            m1_user.append([0] * 128)
    m1_user = np.array(m1_user, dtype=np.float32)
    print(m1_item.shape)
    np.save('./cached_data/m1_item_stage2.npy', m1_item)
    print(m1_user.shape)
    np.save('./cached_data/m1_user_stage2.npy', m1_user)
    ## 所有无点击行为的
    list_df = data.query('label==0').groupby(item_fe_list)['user_f'].apply(lambda x: list(x)).\
        rename('list_user_f').reset_index()
    list_df = setitem_fe_list.merge(list_df, on=item_fe_list, how='left')
    list_df['list_user_f'] = list_df['list_user_f'].map(
        lambda x: x if isinstance(x, list) else ['N'])
    list_df_list = list_df['list_user_f'].values.tolist()
    print('w2v start!')
    model = Word2Vec(
        list_df_list,
        size=128,
        window=10000,  # 增加了10倍
        workers=10,
        min_count=1,
        sg=0,
        hs=0,
        negative=5,
        iter=10,
        seed=0)
    emb_matrix = []
    for seq in list_df_list:  # 对应每个'item_f'
        vec = []
        for w in seq:
            vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * 64)

    emb_matrix = np.array(emb_matrix, dtype=np.float32)
    dict_feature1 = {}
    for index, key in enumerate(list(list_df[item_fe_list])):
        dict_feature1[key] = emb_matrix[index, :]  # feature1的字典 key是item_f
    # 直接存emb
    m1_item = []
    for i in tqdm(list(data[item_fe_list])):
        m1_item.append(dict_feature1[i])
    m1_item = np.array(m1_item, dtype=np.float32)
    m1_user = []
    for i in tqdm(list(data['user_f'])):
        try:
            m1_user.append(model.wv[i])
        except:
            m1_user.append([0] * 128)
    m1_user = np.array(m1_user, dtype=np.float32)
    print(m1_item.shape)
    np.save('./cached_data/m0_item_stage2.npy', m1_item)
    print(m1_user.shape)
    np.save('./cached_data/m0_user_stage2.npy', m1_user)


if __name__ == '__main__':
    print('start!')
    data = Cache.reload_cache('CACHE_sampling_pro_feature.pkl')
    print(data.shape)
    data['label'] = data['label'].fillna(2).astype(int)  # mask
    gc.collect()
    print('w2v start!')
    # 生成一个emb matrix
    user_fe_list = ['age', 'city_rank', 'gender',
                    'slot_id', 'net_type']  # 'city_rank'
    item_fe_list = ['task_id', 'adv_id', 'creat_type_cd', 'dev_id', 'inter_type_cd', 'indu_name', 'adv_prim_id', 'tags', 'spread_app_id',
                    'app_first_class', 'his_on_shelf_time']
    print('join!')  # 简化的预训练方式 将用户属性，广告属性做拼接，大窗口做预训练学习共现分布
    data['user_f'] = ''
    for i, vari in enumerate(user_fe_list):
        data['user_f'] = data['user_f']+_+data[vari].astype('str')
    data['item_f'] = ''
    for i, vari in enumerate(item_fe_list):
        data['item_f'] = data['item_f']+_+data[vari].astype('str')
    gen_list_df(data, 'item_f')
