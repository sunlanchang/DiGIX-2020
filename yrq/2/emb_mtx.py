from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tqdm import tqdm
def get_embedding(path):
        path_kv = path+f'_word2vec_kv.kv'
        wv = KeyedVectors.load(path_kv, mmap='r')
        feature_tokens = list(wv.vocab.keys())
        tokenizer = Tokenizer(num_words=len(feature_tokens)+1)
        tokenizer.fit_on_texts(feature_tokens)
        embedding_dim = 64
        embedding_matrix = np.random.randn(
            len(feature_tokens)+1, embedding_dim)
        for feature in feature_tokens:
            embedding_vector = wv[feature]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([feature])[0][0]
                embedding_matrix[index] = embedding_vector
        print(embedding_matrix.shape)
        return embedding_matrix

if __name__ == '__main__': 
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    for i in tqdm(f1_f2_list):
        embedding_matrix = get_embedding(str(i[0])+'_'+str(i[1]))
        np.save(str(i[0])+'_'+str(i[1])+'_embedding_matrix.npy',embedding_matrix)
    