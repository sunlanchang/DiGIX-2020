# key,value 过去x天时间窗口特征 目前只跑了uid
from multiprocessing import Pool
import pandas as pd
import numpy as np
import gc
from base import Cache
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)


def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    tm_cols = df.select_dtypes('datetime').columns
    for col in df.columns:
        if col in tm_cols:
            continue
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type).find('int') > -1:
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
            elif str(col_type).find('float') > -1:
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


def multi_process_cal(cls, data, params, process_nums=3):
    df_list = []
    key = params['key']  # 根据key分组
    data_group = data.groupby([key])
    step = len(data_group.size()) // process_nums
    index_list = []
    for index, group in enumerate(data_group):
        if index % step != 0:
            index_list.append(group[1])
        elif index % step == 0 and index != 0:
            df_list.append([pd.concat(index_list), params])
            index_list = [group[1]]
        else:
            index_list.append(group[1])
    df_list.append([pd.concat(index_list), params])
    with Pool(process_nums) as p:
        result = p.map(cls, df_list)
    fe = pd.concat(result)
    fe = reduce_mem(fe, use_float16=True)
    return fe


def get_groupby_feature(df_list):
    '''
    # 做by 某个key的过去时间统计特征
    :param df_list: list 参数1 groupby key的数据， 参数2 计算参数
    :return: feature 主键为key
    '''
    # ctr平滑系数
    alpha = 3

    data, params = df_list[0].copy(), df_list[1]
    key = params['key']
    window_list = params['window']  # 过去几天的统计信息[1,8]
    sparse_features = params['sparse_features']
    dense_features = params['dense_features']
    cols = ['index', 'pt_d'] + sparse_features + dense_features
    if key not in cols:
        cols = [key] + cols
    febase = data[cols]
    # data[['index', 'uid', 'task_id', 'adv_id', 'pt_d']]  # .drop_duplicates(subset=['index'])

    sparse_aggfunc = ['count', 'nunique']
    dense_aggfunc = ['max', 'min', 'mean', 'std', 'skew']

    for window in tqdm(window_list):
        # 只算过去一天
        if window >= 1:
            data['pt_d_last'] = data['pt_d'] + window
            count_col = ''
            # by key
            for var in sparse_features:
                for aggf in sparse_aggfunc:
                    if len(count_col) > 0 and aggf == 'count':
                        # 多列只算一次count
                        continue
                    else:
                        # 用户昨天各个sparse 列的统计特征
                        fe = data.groupby([key, 'pt_d_last'])[var].agg([aggf]).rename(
                            columns={aggf: f'{key}_{var}_{window}_{aggf}'}).reset_index()
                        fe.columns = [key, 'pt_d',
                                      f'{key}_{var}_{window}_{aggf}']
                        febase = febase.merge(fe, on=[key, 'pt_d'], how='left')
                        if aggf == 'count':
                            # key昨天曝光次数
                            count_col = f'{key}_{var}_{window}_{aggf}'
                # 多样性
                febase[f'{key}_{var}_{window}_nunique_d_count'] = febase[f'{key}_{var}_{window}_nunique'] / febase[
                    count_col]
            febase[count_col] = febase[count_col].fillna(0)
            for var in dense_features:
                for aggf in dense_aggfunc:
                    # key昨天各个dense 列的统计特征
                    fe = data.groupby([key, 'pt_d_last'])[var].agg([aggf]).rename(
                        columns={aggf: f'{key}_{var}_{window}_{aggf}'}).reset_index()
                    fe.columns = [key, 'pt_d', f'{key}_{var}_{window}_{aggf}']
                    febase = febase.merge(fe, on=[key, 'pt_d'], how='left')

            # 可以补上label 计算过去一天内的ctr
            # 昨天key总点击次数
            fe = data.groupby([key, 'pt_d_last'])['label'].sum().rename(
                f'{key}_clicktimes_{window}').reset_index()
            fe.columns = [key, 'pt_d', f'{key}_clicktimes_{window}']
            febase = febase.merge(fe, on=[key, 'pt_d'], how='left')
            febase[f'{key}_clicktimes_{window}'] = febase[f'{key}_clicktimes_{window}'].fillna(
                0)
            # 用户昨天ctr
            febase[f'{key}_ctr_{window}'] = febase[f'{key}_clicktimes_{window}'] / \
                (febase[count_col] + alpha)

            # 过去一天对今天行样本里的dense_feature的变化特征
            for var in dense_features:
                # 今天的值/昨天的值的均值，今天的值/昨天的值的最大值
                febase[key + '_' + var +
                       f'_d_{window}_mean'] = febase[var] / febase[f'{key}_{var}_{window}_mean']
                febase[key + '_' + var +
                       f'_d_{window}_max'] = febase[var] / febase[f'{key}_{var}_{window}_max']

            # by key,var
            # 过去一天对key对当前的var各个sparse_features的曝光率，点击率，ctr
            for var in sparse_features:
                # 昨天这项的曝光次数
                fe = data.groupby([key, var, 'pt_d_last'])['label'].count().rename(
                    f'{key}_{var}_curr_{window}').reset_index()
                fe.columns = [key, var, 'pt_d', f'{key}_{var}_curr_{window}']
                febase = febase.merge(fe, on=[key, var, 'pt_d'], how='left')
                # 该项曝光占总曝光的比例
                febase[f'{key}_{var}_curr_rate_{window}'] = febase[f'{key}_{var}_curr_{window}'] / (
                    febase[count_col] + alpha)
                # 昨天这项的总点击量
                fe = data.groupby([key, var, 'pt_d_last'])['label'].sum().rename(
                    f'{key}_{var}_clicktimes_{window}').reset_index()
                fe.columns = [key, var, 'pt_d',
                              f'{key}_{var}_clicktimes_{window}']
                febase = febase.merge(fe, on=[key, var, 'pt_d'], how='left')
                # 该项点击占总点击的比例
                febase[f'{key}_{var}_clicktimes_rate_{window}'] = febase[f'{key}_{var}_clicktimes_{window}'] / (
                    febase[f'{key}_clicktimes_{window}'] + alpha)
                # 昨天这项的ctr
                febase[f'{key}_{var}_ctr_{window}'] = febase[f'{key}_{var}_clicktimes_{window}'] / (
                    febase[f'{key}_{var}_curr_{window}'] + alpha)

    for var in ['uid', 'pt_d', 'task_id', 'adv_id'] + sparse_features + dense_features:
        # 留index返回进行merge
        if var in febase.columns:
            del febase[var]
    for var in febase.columns:
        if var.find('count') > -1 or var.find('nunique') > -1 \
                or var.find('times') > -1 or var.find('curr') > -1 or var.find('ctr') > -1:
            febase[var] = febase[var].fillna(0)
    return febase


data = Cache.reload_cache('CACHE_data_0912.pkl')


params = {'key': 'uid',
          'window': [1],
          'sparse_features': ['task_id', 'creat_type_cd', 'adv_id', 'adv_prim_id', 'dev_id',
                              'inter_type_cd', 'spread_app_id', 'tags', 'app_first_class',
                              'app_second_class', 'indu_name', 'slot_id', 'net_type'],
          'dense_features': ['app_score', 'his_app_size', 'his_on_shelf_time', 'device_size']}

features_0 = multi_process_cal(
    get_groupby_feature, data, params, process_nums=40)
features_0 = reduce_mem(features_0, use_float16=False)
Cache.cache_data(features_0, nm_marker='data_step_4_features_0_0913')  # 有index
# params = {'key': 'task_id',
#           'window': [1],
#           'sparse_features': ['uid','age','city','gender','device_name', 'residence','emui_dev',
#                              'slot_id','net_type','consume_purchase','career'],
#           'dense_features': ['city_rank','device_size','list_time','device_price','membership_life_duration',
#                              'communication_avgonline_30d','up_life_duration','up_membership_grade']}
# features_1 = multi_process_cal(get_groupby_feature, data, params, process_nums=50)
# features_1 = reduce_mem(features_1, use_float16=False)
# Cache.cache_data(features_1, nm_marker='data_features_1_0913')
# params = {'key': 'adv_id',
#           'window': [1],
#           'sparse_features': ['uid','age','city','gender','device_name', 'residence','emui_dev',
#                              'slot_id','net_type','consume_purchase','career'],
#           'dense_features': ['city_rank','device_size','list_time','device_price','membership_life_duration',
#                              'communication_avgonline_30d','up_life_duration','up_membership_grade']}
# features_2 = multi_process_cal(get_groupby_feature, data, params, process_nums=50)
# features_2 = reduce_mem(features_2, use_float16=False)
# Cache.cache_data(features_2, nm_marker='data_features_2_0913')
