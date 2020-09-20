import pandas as pd 
import numpy as np
import gc
from base import Cache
from tqdm import tqdm
def write(feature1_feature2):
    list_df = Cache.reload_cache('CACHE_list_df_'+feature1_feature2+'.pkl')[0].values.tolist()
    f = open(feature1_feature2+'.txt', 'w')
    for i in list_df:
        if i:
            for j in i:
                f.write(str(j))
                f.write(' ')
            f.write('\n')
        else:
            f.write(str(-2))
            f.write(' ')
            f.write('\n')
    f.close()
if __name__ == '__main__':
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    for i in tqdm(f1_f2_list):
        write(str(i[0])+'_'+str(i[1]))