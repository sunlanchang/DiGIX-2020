# %%
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import os
import gc
import datetime as dt
import warnings
from tqdm import tqdm
from base import Cache

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)


# %%
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


# %%
##############################################################################################################
# 这块效率好低啊...算了也不想改了，seq='|'应该可以跑通
data_train = pd.read_csv('./data/train_data.csv',
                         sep='|', dtype=str,
                         nrows=100000,
                         )
data_test_A = pd.read_csv('./data/test_data_A.csv',
                          sep='|', dtype=str,
                          nrows=100000,
                          )
data_test_B = pd.read_csv('./data/test_data_B.csv',
                          sep='|', dtype=str,
                          nrows=100000,
                          )
# datatraintestA = Cache.reload_cache('CACHE_dataall0816.pkl')
# datatest = pd.read_csv('./data/test_data_B.csv')
# columns_str = datatest.columns[0] dflisttst = [] for i in tqdm(range(datatest.shape[0])): dflisttst.append([int(j) if index != 32 else j for index, j in enumerate(datatest[columns_str].iloc[i].split('|'))])
# del datatest
# gc.collect()
# dflisttst = pd.DataFrame(dflisttst, columns=columns_str.split('|'))
dataall = pd.concat([data_train, data_test_A, data_test_B], ignore_index=True)
dataall = reduce_mem(dataall, use_float16=False)
# 基础特征+id 希望test a test b的id不重复 日
Cache.cache_data(dataall, nm_marker='dataall_stage2_0924_debug')

# %%
##############################################################################################################
# 比较慢！
# datatraintestA = Cache.reload_cache('CACHE_cmr0816.pkl')
# route = []
# for i in tqdm(range(dataall.shape[0])):
#     route.append(dataall['communication_onlinerate'].iloc[i].split('^'))
# route = pd.DataFrame(route)
# # %%
# route = route.fillna(-1).astype(int)
# routes = []
# for i in tqdm(range(route.shape[0])):
#     routes.append(np.sum(np.eye(25)[route.iloc[i, :]], axis=0))
# del route
# gc.collect()
# # %%
# routes = pd.DataFrame(routes, columns=['cmr_' + str(i)
#                                        for i in range(24)] + ['cmr_None'])
# routes = routes.astype(int)
# routes = reduce_mem(routes, use_float16=False)
# routes = pd.concat([datatraintestA, routes], ignore_index=True)
# Cache.cache_data(routes, nm_marker='cmr_stage2_0924')  # cmr 特征onehott
# 以上为train+test_a+test_b的数据形式


# %%

data = dataall
tokenizer = Tokenizer(num_words=24, filters='^')
communication_onlinerate_dict = [
    '0^1^2^3^4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19^20^21^22^23']
tokenizer.fit_on_texts(communication_onlinerate_dict)

# %%
communication_onlinerate_raw = data['communication_onlinerate'].tolist()
communication_onlinerate_sequences = tokenizer.texts_to_sequences(
    communication_onlinerate_raw)
communication_onlinerate_sequences = pad_sequences(
    communication_onlinerate_sequences, maxlen=24, padding='post')
communication_onlinerate_onehot = []
# %%
with tqdm(total=communication_onlinerate_sequences.shape[0]) as pbar:
    for i in communication_onlinerate_sequences:
        communication_onlinerate_onehot.append(
            np.delete(np.eye(25)[i], 0, axis=1).sum(axis=0))
        pbar.update(1)
# %%
communication_onlinerate_onehot = pd.DataFrame(
    communication_onlinerate_onehot, columns=['cmr_' + str(i) for i in range(24)]).astype(int)
communication_onlinerate_onehot = reduce_mem(
    communication_onlinerate_onehot, use_float16=True)
Cache.cache_data(communication_onlinerate_onehot,
                 nm_marker='cmr_stage2_0924_debug')
print('Train Done')

# %%
