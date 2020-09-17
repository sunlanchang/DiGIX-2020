import pandas as pd 
import numpy as np
import gc
from base import Cache
from tqdm import tqdm

f_path = 'adv_id2'
feature = 'adv_id'
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

train = pd.read_csv(r'train_data.csv', sep='|', dtype=str).drop(columns = ['communication_onlinerate']).astype(int)
train = reduce_mem(train, use_float16=True)
test = pd.read_csv(r'test_data_A.csv', sep='|', dtype=str).drop(columns = ['id','communication_onlinerate']).astype(int)
test.insert(0, 'label', np.ones([1000000]))
test['label'] = 2
test = reduce_mem(test, use_float16=True)
data = pd.concat([train,test],axis=0,ignore_index=True)
data = reduce_mem(data, use_float16=True)   
data_uid_ptd_feature = data[['uid','pt_d',feature]]

list_data = Cache.reload_cache('CACHE_list_df_adv_id.pkl')
list_data.columns=['list','pt_d','uid'] 
list_data = pd.merge(data_uid_ptd_feature,list_data,how='left',on=('uid','pt_d'))
list_data = list_data['list'].values.tolist()
index = 0
list_data_ = []
for i in list_data:
    i.append(data_uid_ptd_feature.at[index, feature])
    list_data_.append(i)
    index = index+1

f = open(f_path+'.txt', 'w')
for i in tqdm(list_data_):
    for j in i:
        f.write(str(j))
        f.write(' ')
    f.write('\n')
f.close()