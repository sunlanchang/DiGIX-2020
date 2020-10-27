import pandas as pd
from utils import reduce_mem

import os
import ipdb
import pickle
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    help='调试模式，只加载一小部分数据',
                    default=False)
args = parser.parse_args()
cached_data_path = 'data/cached_data/train_data.pkl'
original_csv_data_path = 'data/original_csv_data/final/train_data.csv'
if os.path.exists(cached_data_path) and not args.debug:
    print('loading data ...')
    start = time.time()
    with open(cached_data_path, 'rb')as f:
        df = pickle.load(f)
    print('using {:d}s to load data'.format(int(time.time()-start)))
else:
    print('loading original csv file ...')
    df = pd.read_csv(original_csv_data_path,
                     sep='|',
                     nrows=100000 if args.debug else None,
                     )
    df.drop(['communication_onlinerate'], axis=1, inplace=True)
    # 按照uid pt_d升序
    df = df.sort_values(["uid", "pt_d"], ascending=(True, True))
    df = reduce_mem(df)
    if not args.debug:
        with open(cached_data_path, 'wb')as f:
            pickle.dump(df, f)
        print('cache data successful!')

# 两种思路，第一种是只要出现在用户的推荐列表里，无论用户点不点，都用来合成序列
# 第二种是只用用户点击过的作为序列
# 先采用第一种简单处理一下
ipdb.set_trace()

uid_task_id = df.groupby(
    'uid')['task_id'].apply(list).reset_index(name='task_id')

uid_task_id_sequence_path = 'data/feature_sequence/uid_task_id.txt'
with open(uid_task_id_sequence_path, 'w')as f:
    for ids in uid_task_id.task_id:
        ids = [str(e) for e in ids]
        line = ' '.join(ids)
        f.write(line+'\n')
