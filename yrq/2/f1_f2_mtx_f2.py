from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from base import Cache
from tqdm import tqdm
import pandas as pd 
def get_embedding(f1_f2):
    path = f1_f2+'_word2vec_kv.kv'
    wv = KeyedVectors.load(path, mmap='r')
    feature = list(wv.vocab.keys())
    emb_mtx_list = []
    for i in feature:
        buf = []
        buf.append(wv[i].tolist())
        buf.append(i)
        emb_mtx_list.append(buf)
    df_f2_list = pd.DataFrame(emb_mtx_list)
    Cache.cache_data(df_f2_list, nm_marker='list_df_f2_'+f1_f2)
    return 0
if __name__ == '__main__': 
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    for i in tqdm(f1_f2_list):
        get_embedding(str(i[0])+'_'+str(i[1]))
    