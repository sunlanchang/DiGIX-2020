#!/usr/bin/env python
# coding: utf-8

# # data-pkl

# In[1]:


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import pandas as pd
import numpy as np
import gc
from base import Cache
from tqdm import tqdm


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


# In[2]:


# import ipdb
# ipdb.set
train = pd.read_csv(r'./data/train_data.csv', sep='|', dtype=str)
Cache.cache_data(train, nm_marker='train_raw')


# In[3]:


test_A = pd.read_csv(r'./data/test_data_A.csv', sep='|', dtype=str)
test_A.insert(0, 'label', np.ones([1000000]))
test_A['label'] = 2
Cache.cache_data(test_A, nm_marker='test_A_raw')


# In[4]:


test_B = pd.read_csv(r'./data/test_data_B.csv', sep='|', dtype=str)
test_B.insert(0, 'label', np.ones([1000000]))
test_B['label'] = 2
Cache.cache_data(test_B, nm_marker='test_B_raw')


# # cmr-onehot

# In[5]:


tokenizer = Tokenizer(num_words=24, filters='^')
communication_onlinerate_dict = [
    '0^1^2^3^4^5^6^7^8^9^10^11^12^13^14^15^16^17^18^19^20^21^22^23']
tokenizer.fit_on_texts(communication_onlinerate_dict)


# In[6]:


data = Cache.reload_cache('CACHE_train_raw.pkl')
communication_onlinerate_raw = data['communication_onlinerate'].tolist()
communication_onlinerate_sequences = tokenizer.texts_to_sequences(
    communication_onlinerate_raw)
communication_onlinerate_sequences = pad_sequences(
    communication_onlinerate_sequences, maxlen=24, padding='post')
communication_onlinerate_onehot = []
with tqdm(total=communication_onlinerate_sequences.shape[0]) as pbar:
    for i in communication_onlinerate_sequences:
        communication_onlinerate_onehot.append(
            np.delete(np.eye(25)[i], 0, axis=1).sum(axis=0))
        pbar.update(1)
communication_onlinerate_onehot = pd.DataFrame(
    communication_onlinerate_onehot).astype(int)
communication_onlinerate_onehot = reduce_mem(
    communication_onlinerate_onehot, use_float16=True)
Cache.cache_data(communication_onlinerate_onehot, nm_marker='train_cmr_onehot')
print('Train Done')


# In[7]:


data = Cache.reload_cache('CACHE_test_A_raw.pkl')
communication_onlinerate_raw = data['communication_onlinerate'].tolist()
communication_onlinerate_sequences = tokenizer.texts_to_sequences(
    communication_onlinerate_raw)
communication_onlinerate_sequences = pad_sequences(
    communication_onlinerate_sequences, maxlen=24, padding='post')
communication_onlinerate_onehot = []
with tqdm(total=communication_onlinerate_sequences.shape[0]) as pbar:
    for i in communication_onlinerate_sequences:
        communication_onlinerate_onehot.append(
            np.delete(np.eye(25)[i], 0, axis=1).sum(axis=0))
        pbar.update(1)
communication_onlinerate_onehot = pd.DataFrame(
    communication_onlinerate_onehot).astype(int)
communication_onlinerate_onehot = reduce_mem(
    communication_onlinerate_onehot, use_float16=True)
Cache.cache_data(communication_onlinerate_onehot,
                 nm_marker='test_A_cmr_onehot')
print('Test A Done')


# In[8]:


data = Cache.reload_cache('CACHE_test_B_raw.pkl')
communication_onlinerate_raw = data['communication_onlinerate'].tolist()
communication_onlinerate_sequences = tokenizer.texts_to_sequences(
    communication_onlinerate_raw)
communication_onlinerate_sequences = pad_sequences(
    communication_onlinerate_sequences, maxlen=24, padding='post')
communication_onlinerate_onehot = []
with tqdm(total=communication_onlinerate_sequences.shape[0]) as pbar:
    for i in communication_onlinerate_sequences:
        communication_onlinerate_onehot.append(
            np.delete(np.eye(25)[i], 0, axis=1).sum(axis=0))
        pbar.update(1)
communication_onlinerate_onehot = pd.DataFrame(
    communication_onlinerate_onehot).astype(int)
communication_onlinerate_onehot = reduce_mem(
    communication_onlinerate_onehot, use_float16=True)
Cache.cache_data(communication_onlinerate_onehot,
                 nm_marker='test_B_cmr_onehot')
print('Test B Done')


# # concat cmr&raw

# In[9]:


data = Cache.reload_cache('CACHE_train_raw.pkl').drop(
    columns=['communication_onlinerate']).astype(int)
data = reduce_mem(data, use_float16=True)
communication_onlinerate_onehot_data = Cache.reload_cache(
    'CACHE_train_cmr_onehot.pkl')
communication_onlinerate_onehot_data.columns = ['communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                                                'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                                                'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                                                'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                                                'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                                                'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                                                'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                                                'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24']
data = pd.concat([data, communication_onlinerate_onehot_data],
                 axis=1, ignore_index=True)
data.columns = ['label', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class',
                'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size',
                'career', 'gender', 'net_type', 'residence', 'his_app_size',
                'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price',
                'up_life_duration', 'up_membership_grade', 'membership_life_duration',
                'consume_purchase', 'communication_avgonline_30d', 'indu_name', 'pt_d',
                'communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24']
Cache.cache_data(data, nm_marker='train')


# In[10]:


data = Cache.reload_cache('CACHE_test_A_raw.pkl').drop(
    columns=['communication_onlinerate']).astype(int)
data = reduce_mem(data, use_float16=True)
communication_onlinerate_onehot_data = Cache.reload_cache(
    'CACHE_test_A_cmr_onehot.pkl')
communication_onlinerate_onehot_data.columns = ['communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                                                'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                                                'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                                                'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                                                'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                                                'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                                                'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                                                'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24']
data = pd.concat([data, communication_onlinerate_onehot_data],
                 axis=1, ignore_index=True)
data.columns = ['label', 'id', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class',
                'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size',
                'career', 'gender', 'net_type', 'residence', 'his_app_size',
                'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price',
                'up_life_duration', 'up_membership_grade', 'membership_life_duration',
                'consume_purchase', 'communication_avgonline_30d', 'indu_name', 'pt_d',
                'communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24']
Cache.cache_data(data, nm_marker='test_A')


# In[11]:


data = Cache.reload_cache('CACHE_test_B_raw.pkl').drop(
    columns=['communication_onlinerate']).astype(int)
data = reduce_mem(data, use_float16=True)
communication_onlinerate_onehot_data = Cache.reload_cache(
    'CACHE_test_B_cmr_onehot.pkl')
communication_onlinerate_onehot_data.columns = ['communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                                                'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                                                'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                                                'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                                                'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                                                'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                                                'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                                                'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24']
data = pd.concat([data, communication_onlinerate_onehot_data],
                 axis=1, ignore_index=True)
data.columns = ['label', 'id', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class',
                'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size',
                'career', 'gender', 'net_type', 'residence', 'his_app_size',
                'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price',
                'up_life_duration', 'up_membership_grade', 'membership_life_duration',
                'consume_purchase', 'communication_avgonline_30d', 'indu_name', 'pt_d',
                'communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24']
Cache.cache_data(data, nm_marker='test_B')


# In[ ]:
