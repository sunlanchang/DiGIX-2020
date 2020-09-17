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

def gen_list_df(data,feature1,feature2):
    try:
        data_group = data.groupby([feature1])
        del data
        gc.collect()
        feature2_name_list = []
        for name,group in data_group:
            feature2_name_list.append(name)    
        list_feature2 = []
        for i in feature2_name_list:
            list_feature2_ = []
            index_get_group = data_group.get_group(i)
            buf = []
            for j in index_get_group[[feature2,'label']].values:
                if j[1] == 1:
                    buf.append(j[0])
            list_feature2_.append(buf)
            list_feature2_.append(i)
            list_feature2.append(list_feature2_)
        list_df = pd.DataFrame(list_feature2)
        Cache.cache_data(list_df, nm_marker='list_df_'+feature1+'_'+feature2)
        del list_df,data_group,feature2_name_list,list_feature2_,index_get_group,list_feature2
        gc.collect()
        return True
    except:
        return False

if __name__ == '__main__':
    train = pd.read_csv(r'train_data.csv', sep='|', dtype=str).drop(columns = ['communication_onlinerate']).astype(int)
    train = reduce_mem(train, use_float16=True)
    test = pd.read_csv(r'test_data_A.csv', sep='|', dtype=str).drop(columns = ['id','communication_onlinerate']).astype(int)
    test.insert(0, 'label', np.ones([1000000]))
    test['label'] = 2
    test = reduce_mem(test, use_float16=True)
    data = pd.concat([train,test],axis=0,ignore_index=True)
    data = reduce_mem(data, use_float16=True)
    del train,test
    gc.collect()
    poc_feature1_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence']]
    for i in poc_feature1_list:
        if gen_list_df(data,i[0],i[1]):
            print(i,' Done')
        else:
            print(i,' Err')

