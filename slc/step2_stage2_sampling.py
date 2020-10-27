#!/usr/bin/env python
# coding: utf-8

# # Sampling

# In[3]:


import pandas as pd
import numpy as np
from base import Cache
## 按天采样 保留比例1:5 1:5.5 都做了后续的模型


def get_sample(df, day, neg_rate=5):
    set1 = df.query('pt_d=={}'.format(day))
    set1_pos = set1.query('label==1')
    nums_pos = set1_pos.shape[0]
    nums_neg = nums_pos * neg_rate
    set1_neg = set1.query('label==0')
    set1_neg = set1_neg.sample(nums_neg, random_state=0)
    df_sample = pd.concat([set1_pos, set1_neg])
    print(df_sample['label'].value_counts(), df_sample['label'].mean())
    return df_sample


# In[4]:


train = Cache.reload_cache('CACHE_train.pkl')
train = train.reset_index()
train.rename(columns={'index': 'raw_index'}, inplace=True)

test_B = Cache.reload_cache('CACHE_test_B.pkl').drop(columns=['id'])
test_B = test_B.reset_index()
test_B.rename(columns={'index': 'raw_index'}, inplace=True)
test_B['raw_index'] = test_B['raw_index']+41907133

train_ptd_1 = get_sample(train, 1)
train_ptd_2 = get_sample(train, 2)
train_ptd_3 = get_sample(train, 3)
train_ptd_4 = get_sample(train, 4)
train_ptd_5 = get_sample(train, 5)
train_ptd_6 = get_sample(train, 6)
train_ptd_7 = get_sample(train, 7)

train_sampling = pd.concat([train_ptd_1, train_ptd_2], ignore_index=True)
train_sampling = pd.concat([train_sampling, train_ptd_3], ignore_index=True)
train_sampling = pd.concat([train_sampling, train_ptd_4], ignore_index=True)
train_sampling = pd.concat([train_sampling, train_ptd_5], ignore_index=True)
train_sampling = pd.concat([train_sampling, train_ptd_6], ignore_index=True)
train_sampling = pd.concat([train_sampling, train_ptd_7], ignore_index=True)

Cache.cache_data(train_sampling, nm_marker='train_sampling_pos1_neg5')

sampling_data = pd.concat([train_sampling, test_B], ignore_index=True)
Cache.cache_data(sampling_data, nm_marker='data_sampling_pos1_neg5')


# ## 填充缺失值部分也有尝试作为子模型 填充方式如下：

# In[ ]:


## 修正一些异常值 增加模型鲁棒性
data = train_sampling
# 修正缺失值
sparse_features = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd',
                   'slot_id', 'spread_app_id', 'tags', 'app_first_class',
                   'app_second_class', 'city', 'device_name', 'career', 'gender',
                   'net_type', 'residence', 'emui_dev', 'indu_name', 'communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
                   'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
                   'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
                   'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
                   'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
                   'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
                   'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
                   'communication_onlinerate_22', 'communication_onlinerate_23', 'communication_onlinerate_24', 'age', 'city_rank']
dense_features = ['his_app_size', 'his_on_shelf_time', 'app_score', 'device_size', 'list_time', 'device_price', 'up_life_duration',
                  'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_avgonline_30d']
## 认为出现-1是缺失值，sparse feature填众数 dense feature填平均值
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

Cache.cache_data(data, nm_marker='train_sampling_pos1_neg5')
