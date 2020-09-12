import pandas as pd
import numpy as np
import gc
from base import Cache
from tqdm import tqdm

def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
    tm_cols = df.select_dtypes('datetime').columns
    colsuse = [i for i in df.columns if i!= 'label']
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
	
##############################################################################################################
datatrain = pd.read_csv('train_data.csv')
datatest = pd.read_csv('test_data_A.csv')
columns_str = datatrain.columns[0]
gc.collect()
dflist = []
for i in tqdm(range(datatrain.shape[0])):
    dflist.append([ int(j) if index!=32 else j for index,j in enumerate(datatrain[columns_str].iloc[i].split('|'))])
dflist = pd.DataFrame(dflist,columns = columns_str.split('|'))
del datatrain
gc.collect()
columns_str = datatest.columns[0]
dflisttst = []
for i in tqdm(range(datatest.shape[0])):
    dflisttst.append([ int(j) if index!=32 else j for index,j in enumerate(datatest[columns_str].iloc[i].split('|'))])
del datatest
gc.collect()
dflisttst = pd.DataFrame(dflisttst,columns = columns_str.split('|'))
dflist['id'] = -1# train id都改成-1
dataall = pd.concat([dflist,dflisttst],ignore_index=True)
del dflist,dflisttst
gc.collect()
dataall = reduce_mem(dataall, use_float16=False)
Cache.cache_data(dataall, nm_marker='dataall0816')

##############################################################################################################
# 比较慢！
route = []
for i in tqdm(range(dataall.shape[0])):
    route.append(dataall['communication_onlinerate'].iloc[i].split('^'))
route = pd.DataFrame(route)
route = route.fillna(-1).astype(int)
routes = []
for i in tqdm(range(route.shape[0])):
    routes.append(np.sum(np.eye(25)[route.iloc[i,:]],axis=0))
del route
gc.collect()
routes = pd.DataFrame(routes,columns=['cmr_'+str(i) for i in range(24)]+['cmr_None'])
routes = routes.astype(int)
routes = reduce_mem(routes, use_float16=False)
Cache.cache_data(routes, nm_marker='cmr0816')
del dataall,routes
gc.collect()
# ##############################################################################################################

# data = Cache.reload_cache('CACHE_dataall0816.pkl')
# print(data.dtypes)
# data['communication_onlinerate'] = data['communication_onlinerate'].map(lambda x:x.replace('^',' '))
# route = Cache.reload_cache('CACHE_cmr0816.pkl')
# data = pd.concat([data,route],axis=1)
# print(data.dtypes)
# del route
# gc.collect()
# # 规整
# data.loc[data['adv_prim_id']==101,'adv_prim_id']=-1
# data.loc[data['dev_id']<13,'dev_id']=-1
# data.loc[data['dev_id']>71,'dev_id']=-1
# data.loc[data['spread_app_id']>88,'spread_app_id']=-1
# data.loc[data['tags']>88,'tags']=-1
# data.loc[data['device_name']<12,'device_name']=-1
# data.loc[data['indu_name']>51,'indu_name']=51
# var_unk_list = ['adv_prim_id',
# 'dev_id',
# 'spread_app_id',
# 'tags',
# 'app_second_class',
# 'device_name',
# 'emui_dev',
# 'indu_name']
# for var in var_unk_list:
#     print('process unk: ',var)
#     s1 = set(data.query('label == label')[var])
#     s2 = set(data.query('label != label')[var])
#     setuse = s1.intersection(s2)
#     data.loc[~data[var].isin(setuse),var]=-1# unk
#     data[var] = data[var].astype(int)
# # uid 过去的行为
# data = data.sort_values(['uid','pt_d'],ascending=False)
# var_diff_list = ['device_size','his_app_size','his_on_shelf_time','app_score','list_time','device_price',
#                  'up_life_duration','up_membership_grade','membership_life_duration','consume_purchase','communication_avgonline_30d']
# for var in var_diff_list:
#     print('process diff: ',var)
#     data[var+'_diff'] = data.groupby('uid')[var].diff(-1)
#     data[var+'_shift_1'] = data.groupby('uid')[var].shift(-1)
#     data[var+'_shift_2'] = data.groupby('uid')[var].shift(-2)
#     data[var+'_shift_3'] = data.groupby('uid')[var].shift(-3)

# sparse_fe = ['net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career',
#              'city','consume_purchase','uid','dev_id','tags','slot_id']
# for var in sparse_fe:
#     print('process int: ',var)
#     data[var] = data[var].fillna(0)
#     data[var] = data[var].astype(int)
# data = reduce_mem(data, use_float16=False)

# # 开源特征
# ##########################cate feature#######################
# cate_cols = ['slot_id','net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','dev_id','tags']
# for f in tqdm(cate_cols):
#     map_dict = dict(zip(data[f].unique(), range(data[f].nunique())))
#     data[f + '_count'] = data[f].map(data[f].value_counts())

# ##########################groupby feature#######################
# def group_fea(data,key,target):
#     tmp = data.groupby(key, as_index=False)[target].agg({
#         key+target + '_nunique': 'nunique',
#     }).reset_index()
#     del tmp['index']
#     print("**************************{}**************************".format(target))
#     return tmp

# feature_key = ['uid','age','career','net_type']
# feature_target = ['task_id','adv_id','dev_id','slot_id','spread_app_id','indu_name']

# for key in tqdm(feature_key):
#     for target in feature_target:
#         tmp = group_fea(data,key,target)
#         data = data.merge(tmp,on=key,how='left')
# del tmp
# gc.collect()
# data = reduce_mem(data, use_float16=False)

# test_df = data[data["pt_d"]==8].copy().reset_index()
# train_df = data[data["pt_d"]<8].reset_index()
# del data
# gc.collect()

# #统计做了groupby特征的特征
# group_list = []
# for s in train_df.columns:
#     if '_nunique' in s:
#         group_list.append(s)
# print(group_list)


# ##########################target_enc feature#######################
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
# enc_list = group_list + ['net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','uid_count','dev_id','tags','slot_id']
# for f in tqdm(enc_list):
#     train_df[f + '_target_enc'] = 0
#     test_df[f + '_target_enc'] = 0
#     for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
#         trn_x = train_df[[f, 'label']].iloc[trn_idx].reset_index(drop=True)
#         val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)
#         enc_df = trn_x.groupby(f, as_index=False)['label'].agg({f + '_target_enc': 'mean'})
#         val_x = val_x.merge(enc_df, on=f, how='left')
#         test_x = test_df[[f]].merge(enc_df, on=f, how='left')
#         val_x[f + '_target_enc'] = val_x[f + '_target_enc'].fillna(train_df['label'].mean())
#         test_x[f + '_target_enc'] = test_x[f + '_target_enc'].fillna(train_df['label'].mean())
#         train_df.loc[val_idx, f + '_target_enc'] = val_x[f + '_target_enc'].values
#         test_df[f + '_target_enc'] += test_x[f + '_target_enc'].values / skf.n_splits
        
# del trn_x,val_x,enc_df,test_x
# gc.collect()
# # all features
# df_fe = pd.concat([train_df,test_df])
# del train_df,test_df
# df_fe = reduce_mem(df_fe, use_float16=False)
# Cache.cache_data(df_fe , nm_marker='datafeature0816')