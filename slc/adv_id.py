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
                     nrows=1000000 if args.debug else None,
                     )
    df.drop(['communication_onlinerate'], axis=1, inplace=True)
    df = reduce_mem(df)
    if not args.debug:
        with open(cached_data_path, 'wb')as f:
            pickle.dump(df, f)
        print('cache data successful!')

uid_task_id = df.groupby(
    'uid')['task_id'].apply(list).reset_index(name='task_id')
ipdb.set_trace()
