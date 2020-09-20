# 基础特征+count+target encoding
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool
import pandas as pd
import numpy as np
import gc
from base import Cache
from tqdm import tqdm


def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
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
    end_mem = df.memory_usage().sum() / 1024**2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


data = Cache.reload_cache('CACHE_dataall0816.pkl')
print(data.dtypes)
data['communication_onlinerate'] = data['communication_onlinerate'].map(
    lambda x: x.replace('^', ' '))
route = Cache.reload_cache('CACHE_cmr0816.pkl')
route_columns = [i for i in route.columns]
data = pd.concat([data, route], axis=1)  # 无index
data = data.reset_index(drop=True).reset_index()
# 计算每个样本当日重复几次
cols = [i for i in data.columns if i not in ['index', 'id']]
fe = data.groupby(cols)['index'].count().rename('sample_nunique').reset_index()
fe_max = fe.groupby('pt_d')['sample_nunique'].max().rename(
    'sample_nunique_max').reset_index()
fe = fe.merge(fe_max, on='pt_d', how='left')
fe['sample_nunique'] = fe['sample_nunique']/fe['sample_nunique_max']
del fe['sample_nunique_max']
data = data.merge(fe, on=cols, how='left')  # 保留相对值
gc.collect()

data1 = data.query('pt_d<8').drop_duplicates(subset=cols)  # 重复样本去掉
data2 = data.query('pt_d==8')
data = pd.concat([data1, data2], ignore_index=True)
data = data.sort_values(['uid', 'pt_d', 'slot_id', 'net_type',
                         'task_id', 'adv_id'], ascending=False).reset_index(drop=True)
del data1, data2
gc.collect()

for var in route_columns:
    data[var] = data[var].astype(int)
print(data.dtypes)
del route
gc.collect()

# 修正缺失值
sparse_features = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career', 'gender', 'net_type', 'residence', 'emui_dev',
                   'indu_name', 'cmr_0', 'cmr_1', 'cmr_2', 'cmr_3', 'cmr_4', 'cmr_5', 'cmr_6', 'cmr_7', 'cmr_8', 'cmr_9', 'cmr_10', 'cmr_11', 'cmr_12', 'cmr_13', 'cmr_14', 'cmr_15', 'cmr_16', 'cmr_17', 'cmr_18', 'cmr_19', 'cmr_20', 'cmr_21', 'cmr_22', 'cmr_23', 'age', 'city_rank']
dense_features = ['his_app_size', 'his_on_shelf_time', 'app_score', 'device_size', 'list_time', 'device_price', 'up_life_duration',
                  'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_avgonline_30d', 'cmr_None', 'sample_nunique']

for var in sparse_features:
    mode_num = data[var].mode()[0]
    shape_null = data.query('{}==-1'.format(var)).shape[0]
    print('process sparse int: ', var, 'fillna: ',
          mode_num, 'fillna_shape: ', shape_null)
    if shape_null > 0:
        data.loc[data[var] == -1, var] = mode_num
        data[var] = data[var].astype(int)

for var in dense_features:
    mode_num = int(data[var].mean())
    shape_null = data.query('{}==-1'.format(var)).shape[0]
    print('process dense int: ', var, 'fillna: ',
          mode_num, 'fillna_shape: ', shape_null)
    if shape_null > 0:
        data.loc[data[var] == -1, var] = mode_num
        data[var] = data[var].astype(int)
data = reduce_mem(data, use_float16=False)
Cache.cache_data(data, nm_marker='data_0912')  # basefeature


def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
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
    end_mem = df.memory_usage().sum() / 1024**2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


data = Cache.reload_cache('CACHE_data_0912.pkl')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)
# 提取相对count特征
# 列并行

cate_cols = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career', 'gender', 'net_type', 'residence', 'emui_dev',
             'indu_name', 'cmr_0', 'cmr_1', 'cmr_2', 'cmr_3', 'cmr_4', 'cmr_5', 'cmr_6', 'cmr_7', 'cmr_8', 'cmr_9', 'cmr_10', 'cmr_11', 'cmr_12', 'cmr_13', 'cmr_14', 'cmr_15', 'cmr_16', 'cmr_17', 'cmr_18', 'cmr_19', 'cmr_20', 'cmr_21', 'cmr_22', 'cmr_23', 'age']
cate_cols_df = []
for var in tqdm(cate_cols):
    cate_cols_df.append(data[['uid', 'pt_d', var]])


def cls(df):
    f = df.columns[-1]
    mapping = dict(df.query('pt_d<8')[f].value_counts(
    )/df.query('pt_d<8')[f].value_counts().max())  # 只统计train
    mapping_test = dict(df.query('pt_d==8')[f].value_counts(
    )/df.query('pt_d==8')[f].value_counts().max())  # 只统计test
    for key, value in mapping_test.items():
        # 优先用train
        if key not in mapping:
            mapping[key] = value
    df[f + '_count'] = df[f].map(mapping)  # 映射
    fe = df.groupby([f, 'pt_d'])['uid'].count().rename(
        f'{f}_pt_d_count').reset_index()  # 当天统计count
    fe_max = fe.groupby('pt_d')[f'{f}_pt_d_count'].max().rename(
        f'{f}_pt_d_count_max').reset_index()
    fe = fe.merge(fe_max, on='pt_d', how='left')
    fe[f'{f}_pt_d_count'] = fe[f'{f}_pt_d_count']/fe[f'{f}_pt_d_count_max']
    fe[f'{f}_pt_d_count'] = fe[f'{f}_pt_d_count'].fillna(0)
    del fe[f'{f}_pt_d_count_max']
    df = df.merge(fe, on=[f, 'pt_d'], how='left')
    print(df.columns)
    return df[[f, 'pt_d', f + '_count', f'{f}_pt_d_count']]


with Pool(8) as p:
    result = p.map(cls, cate_cols_df)
for index, fe in enumerate(result):
    f = cate_cols[index]
    data = pd.concat([data, fe[fe.columns[-2:]]], axis=1)
    print(fe.columns[-2:], f, data.shape)
    del fe
    gc.collect()
del result, f, cate_cols_df
gc.collect()
data = reduce_mem(data, use_float16=False)

##########################groupby feature#######################


def group_fea(data, key, target):
    tmp = data.groupby(key, as_index=False)[target].agg({
        key+target + '_nunique': 'nunique',
    }).reset_index()
    del tmp['index']
    return tmp


def group_fea_pt_d(data, key, target):
    tmp = data.groupby([key, 'pt_d'], as_index=False)[target].agg({
        key+target + '_pt_d_nunique': 'nunique',
    }).reset_index()
    fe = tmp.groupby('pt_d')[key+target +
                             '_pt_d_nunique'].max().rename('dmax').reset_index()
    tmp = tmp.merge(fe, on='pt_d', how='left')
    tmp[key+target + '_pt_d_nunique'] = tmp[key +
                                            target + '_pt_d_nunique']/tmp['dmax']
    del tmp['index'], tmp['dmax']
    print("**************************{}**************************".format(target))
    return tmp


feature_key = ['uid', 'age', 'gender', 'career', 'city', 'slot_id', 'net_type']
feature_target = ['task_id', 'adv_id', 'dev_id', 'spread_app_id', 'indu_name']

for key in tqdm(feature_key):
    for target in feature_target:
        tmp = group_fea(data, key, target)
        data = data.merge(tmp, on=key, how='left')
        tmp = group_fea_pt_d(data, key, target)
        data = data.merge(tmp, on=[key, 'pt_d'], how='left')
del tmp
gc.collect()
data = reduce_mem(data, use_float16=False)

test_df = data[data["pt_d"] == 8].copy().reset_index()
train_df = data[data["pt_d"] < 8].reset_index()
del data
gc.collect()

# 统计做了groupby特征的特征
group_list = []
for s in train_df.columns:
    if '_nunique' in s:
        group_list.append(s)
print(group_list)


##########################target_enc feature#######################
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
enc_list = group_list + ['net_type', 'task_id', 'adv_id', 'adv_prim_id', 'age',
                         'app_first_class', 'app_second_class', 'career', 'city', 'consume_purchase', 'uid', 'dev_id', 'tags', 'slot_id']
for f in tqdm(enc_list):
    train_df[f + '_target_enc'] = 0
    test_df[f + '_target_enc'] = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        trn_x = train_df[[f, 'label']].iloc[trn_idx].reset_index(drop=True)
        val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)[
            'label'].agg({f + '_target_enc': 'mean'})
        val_x = val_x.merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        val_x[f + '_target_enc'] = val_x[f +
                                         '_target_enc'].fillna(train_df['label'].mean())
        test_x[f + '_target_enc'] = test_x[f +
                                           '_target_enc'].fillna(train_df['label'].mean())
        train_df.loc[val_idx, f +
                     '_target_enc'] = val_x[f + '_target_enc'].values
        test_df[f + '_target_enc'] += test_x[f +
                                             '_target_enc'].values / skf.n_splits

del trn_x, val_x, enc_df, test_x
gc.collect()
# all features
df_fe = pd.concat([train_df, test_df])
del train_df, test_df
df_fe = df_fe.sort_values('index').reset_index(drop=True)
df_fe = reduce_mem(df_fe, use_float16=False)
droplist = []
set_test = df_fe.query('pt_d==8')
for var in df_fe.columns:
    if var not in ['id', 'index', 'label', 'pt_d']:
        if set_test[var].nunique() < 2 or set_test[var].count() < 2:
            droplist.append(var)
print('drop list:', droplist)
df_fe = df_fe.drop(droplist, axis=1)
Cache.cache_data(df_fe, nm_marker='data_step_2_0913')
