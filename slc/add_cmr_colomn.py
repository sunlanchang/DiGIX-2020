import pandas as pd
import numpy as np
from tqdm import tqdm

# def get_train_test_csv():
df_train = pd.read_csv('data/train_data.csv', sep='|', dtype=str)
df_test = pd.read_csv('data/test_data_A.csv', sep='|', dtype=str)
df_train = df_train.drop(columns=['label'])
#     for name in df_train.columns:
#         df_train[name] = df_train[name] + name

df_test = df_test.drop(columns=['id'])
#     for name in df_test.columns:
#         df_test[name] = df_test[name] + name
data = pd.concat([df_train, df_test], ignore_index=True)
# data.to_csv('data/train_test.csv', sep=',', index=False, header=True)
# get_train_test_csv()
# communication_onlinerate 列做onehot
route = []
for i in tqdm(range(data.shape[0])):
    route.append(data['communication_onlinerate'].iloc[i].split('^'))
route = pd.DataFrame(route)

route = route.fillna(-1).astype(int)
routes = []
for i in tqdm(range(route.shape[0])):
    routes.append(np.sum(np.eye(25)[route.iloc[i, :]], axis=0))

routes = pd.DataFrame(routes, columns=['cmr_'+str(i)
                                       for i in range(24)]+['cmr_None'])
data = data.drop(columns=['communication_onlinerate'])
data = pd.concat([data, routes], ignore_index=True, axis=1)
data = data.astype(int)
data.to_pickle('data/train_test.pkl')
