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

def gen_list_df(feature):
    try:
        train = pd.read_csv(r'train_data.csv', sep='|', dtype=str).drop(columns = ['communication_onlinerate']).astype(int)
        train = reduce_mem(train, use_float16=True)

        test = pd.read_csv(r'test_data_A.csv', sep='|', dtype=str).drop(columns = ['id','communication_onlinerate']).astype(int)
        test.insert(0, 'label', np.ones([1000000]))
        test['label'] = 2
        test = reduce_mem(test, use_float16=True)

        data = pd.concat([train,test],axis=0,ignore_index=True)
        data = reduce_mem(data, use_float16=True)
        data_group = data.groupby(['uid'])
        del train,test,data
        gc.collect()
        index_list = []
        feature_list = []
        print('index_list start')
        for name,group in tqdm(data_group):
            index_list.append(name)    
        print('feature_list start')
        for i in tqdm(index_list):
            index_get_group = data_group.get_group(i)
            ptd_set = set(index_get_group['pt_d'].values.flatten().tolist())
            for j in ptd_set:
                feature_list_ = []
                buf_list = []
                buf_list = index_get_group.query('pt_d < @j')[feature].values.flatten().tolist()
                buf_list.append(2)
                feature_list_.append(buf_list)
                feature_list_.append(j)
                feature_list_.append(i)
                feature_list.append(feature_list_)

        list_df = pd.DataFrame(feature_list)
        Cache.cache_data(list_df, nm_marker='list_df_2'+feature)
        del index_list,feature_list,feature_list_,list_df,data_group,index_get_group,ptd_set
        gc.collect()
        return True
    except:
        return False

if __name__ == '__main__':
    poc_feature_list = ['label']#'task_id','adv_id','dev_id','inter_type_cd','spread_app_id','tags','app_first_class','app_second_class','his_app_size','his_on_shelf_time','app_score',,'creat_type_cd','adv_prim_id','indu_name'

    for i in poc_feature_list:
        if gen_list_df(i):
            print(i,' Done')
        else:
            print(i,' Err')

