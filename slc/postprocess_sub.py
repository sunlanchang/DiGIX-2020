#%%
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import pdb
sub = pd.read_csv('data/submission_nn_zlh_10folds.csv')
sub.probability = sub.probability * 10
# %%
test = pd.read_csv('data/test_data_B.csv', sep='|', dtype=str)
train = pd.read_csv('data/train_data.csv', sep='|', dtype=str)
train.label = train.label.astype(int)
# %%
# tmp = train.groupby(['uid', 'adv_id', 'spread_app_id','slot_id', 'net_type']).size()
unique_train = train.groupby(
    ['uid', 'adv_id', 'spread_app_id', 'slot_id', 'net_type']).first()
#%%
# tmp = train.groupby(['uid', 'adv_id', 'spread_app_id',
#  'slot_id', 'net_type']).agg(lambda x: x.value_counts().index[0])
# %%
# %%
test_label = pd.merge(test, unique_train, how='left', validate='m:1',
                      on=['uid', 'adv_id', 'spread_app_id', 'slot_id', 'net_type'])
# %%
test_label = pickle.load(open('cached_data/tmp.pkl', 'rb'))
# %%
print(sub.probability.mean())
for i in tqdm(range(sub.shape[0])):
    if not np.isnan(test_label.iloc[i].label):
        sub.loc[i, 'probability'] = test_label.iloc[i].label
print(sub.probability.mean())
# pdb.set_trace()
# %%
# train_test = pd.merge(test, train, how='inner', on=[
#   'uid', 'adv_id', 'spread_app_id', 'slot_id', 'net_type'])
# %%
# train_test.label = train_test.label.astype(int)
# train_test.groupby(['uid', 'adv_id', 'spread_app_id',
# 'slot_id', 'net_type']).label.agg('max')
# %%
# train_test.groupby(['uid', 'adv_id', 'spread_app_id',
# 'slot_id', 'net_type']).size().min()
# %%
# sub.query('probability<=0.18').shape[0]/sub.shape[0] * 100

# %%
# sub.loc[sub.probability <= 0.18, 'probability'] = 0
# %%
sub.to_csv('data/submission.csv', header=True, index=False)

#%%
sub.id.nunique()
