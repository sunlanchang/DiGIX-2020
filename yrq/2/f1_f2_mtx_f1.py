from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from base import Cache
from tqdm import tqdm
import pandas as pd 
def get_embedding(f1_f2,f1):
    path = f1_f2+'_word2vec_kv.kv'
    wv = KeyedVectors.load(path, mmap='r')
    list_df = Cache.reload_cache('CACHE_list_df_'+f1_f2+'.pkl')
    list_df.columns=['list',f1] 
    f = open(f1_f2+'.txt','r')
    ind = 0
    buf = []
    for i in f:
        buf_ = np.zeros(64)
        for j in i.strip().split(' '):
            buf_ = buf_+wv[j]
        buf_ = buf_/len(i)
        buf_f1 = list_df.at[ind, f1]
        buf__ = []
        buf_ = buf_.tolist()
        buf__.append(buf_)
        buf__.append(buf_f1)
        buf.append(buf__)
        ind = ind+1
    df_f1_list = pd.DataFrame(buf) 
    Cache.cache_data(df_f1_list, nm_marker='list_df_avg_'+f1_f2)
    return 0

if __name__ == '__main__': 
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    for i in tqdm(f1_f2_list):
        get_embedding(str(i[0])+'_'+str(i[1]),i[0])
    