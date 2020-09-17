from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from tqdm import tqdm
from base import Cache
def input_w2v(f1_f2,all_data,f1):
    feature_seq = all_data[[f1]].values.flatten().astype(str).tolist()

    avg_f1 = Cache.reload_cache('CACHE_list_df_avg_'+f1_f2+'.pkl')
    feature_tokens = avg_f1[[1]].values.flatten().astype(str).tolist()
    tokenizer = Tokenizer(num_words=len(feature_tokens)+1)
    tokenizer.fit_on_texts(feature_tokens)
    
    npy_path = f1_f2
    sequences = tokenizer.texts_to_sequences(feature_seq[:41907133])
    x_train = pad_sequences(
        sequences, maxlen=1, padding='post')
    print(x_train.shape)
    np.save(npy_path+'_f1_train.npy', x_train)

    sequences = tokenizer.texts_to_sequences(feature_seq[41907133:])
    x_test = pad_sequences(
        sequences, maxlen=1, padding='post') 
    print(x_test.shape)
    np.save(npy_path+'_f1_test.npy', x_test)

if __name__ == '__main__':
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    all_data = Cache.reload_cache('CACHE_data_deepfm.pkl')
    for i in tqdm(f1_f2_list):
        input_w2v(str(i[0])+'_'+str(i[1]),all_data,str(i[0]))
    
    

    
    