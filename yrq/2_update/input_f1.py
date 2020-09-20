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
    f1_f2_list = [['task_id','age_slot_id'],['task_id','city_slot_id'],['task_id','city_rank_slot_id'],['task_id','device_name_slot_id'],['task_id','career_slot_id'],['task_id','gender_slot_id'],['task_id','residence_slot_id'],['adv_id','age_slot_id'],['adv_id','city_slot_id'],['adv_id','city_rank_slot_id'],['adv_id','device_name_slot_id'],['adv_id','career_slot_id'],['adv_id','gender_slot_id'],['adv_id','residence_slot_id'],['creat_type_cd','age_slot_id'],['creat_type_cd','city_slot_id'],['creat_type_cd','city_rank_slot_id'],['creat_type_cd','device_name_slot_id'],['creat_type_cd','career_slot_id'],['creat_type_cd','gender_slot_id'],['creat_type_cd','residence_slot_id'],['indu_name','age_slot_id'],['indu_name','city_slot_id'],['indu_name','city_rank_slot_id'],['indu_name','device_name_slot_id'],['indu_name','career_slot_id'],['indu_name','gender_slot_id'],['indu_name','residence_slot_id'],['adv_prim_id','age_slot_id'],['adv_prim_id','city_slot_id'],['adv_prim_id','city_rank_slot_id'],['adv_prim_id','device_name_slot_id'],['adv_prim_id','career_slot_id'],['adv_prim_id','gender_slot_id'],['adv_prim_id','residence_slot_id']]
    all_data = Cache.reload_cache('CACHE_data_deepfm.pkl')
    all_data['age_slot_id'] = all_data['age']+all_data['slot_id']
    all_data['city_slot_id'] = all_data['city']+all_data['slot_id']
    all_data['city_rank_slot_id'] = all_data['city_rank']+all_data['slot_id']
    all_data['device_name_slot_id'] = all_data['device_name']+all_data['slot_id']
    all_data['career_slot_id'] = all_data['career']+all_data['slot_id']
    all_data['gender_slot_id'] = all_data['gender']+all_data['slot_id']
    all_data['residence_slot_id'] = all_data['residence']+all_data['slot_id']
    for i in tqdm(f1_f2_list):
        input_w2v(str(i[0])+'_'+str(i[1]),all_data,str(i[0]))
    
    

    
    