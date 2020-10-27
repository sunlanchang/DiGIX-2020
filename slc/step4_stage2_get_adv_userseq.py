#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import gc
from base import Cache
from tqdm import tqdm


# In[2]:


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


# # list_df

# In[3]:



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
        Cache.cache_data(list_df, nm_marker='list_df_adv_userseq_'+feature1+'_'+feature2)
        del list_df,data_group,feature2_name_list,list_feature2_,index_get_group,list_feature2
        gc.collect()
        return True
    except:
        return False


# In[4]:



train = Cache.reload_cache('CACHE_train_raw.pkl').drop(columns = ['communication_onlinerate']).astype(int)
train = reduce_mem(train, use_float16=True)
test = Cache.reload_cache('CACHE_test_B_raw.pkl').drop(columns = ['id','communication_onlinerate']).astype(int)
test = reduce_mem(test, use_float16=True)
data = pd.concat([train,test],axis=0,ignore_index=True)
data = reduce_mem(data, use_float16=True)
del train,test
gc.collect()
poc_feature1_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],
                  ['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],
                  ['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],
                  ['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],
                  ['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],
                  ['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],
                  ['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],
                  ['adv_prim_id','residence']]
for i in tqdm(poc_feature1_list):
    if gen_list_df(data,i[0],i[1]):
        print(i,' Done')
    else:
        print(i,' Err')


# # list_txt

# In[5]:


def write(feature1_feature2):
    list_df = Cache.reload_cache('CACHE_list_df_adv_userseq_'+feature1_feature2+'.pkl')[0].values.tolist()
    f = open('adv_userseq_'+feature1_feature2+'.txt', 'w')
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


# In[6]:


f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],
              ['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],
              ['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],
              ['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],
              ['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],
              ['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],
              ['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],
              ['adv_prim_id','residence']]
for i in tqdm(f1_f2_list):
    write(str(i[0])+'_'+str(i[1]))


# # w2v

# In[7]:


from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


# In[8]:


def f1_f2_w2v(f1_f2):
    print('LineSentence start')
    sentences = word2vec.LineSentence('adv_userseq_'+f1_f2+'.txt') 
    print('Word2Vec start')
    # model = word2vec.Word2Vec(sentences , size=128, window=35, sg=0, hs=1, min_count=1, iter=5, callbacks=[bsz_loss])
    model = word2vec.Word2Vec(sentences , size=64, window=10000, sg=0, hs=1, min_count=1, iter=10, workers=-1)
    print('save start')
    model.save('adv_userseq_'+f1_f2+'_word2vec.model')
    model.wv.save('adv_userseq_'+f1_f2+'_word2vec.kv')
    print('Done')


# In[9]:


f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],
              ['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],
              ['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],
              ['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],
              ['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],
              ['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],
              ['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
for i in f1_f2_list:
    f1_f2_w2v(str(i[0])+'_'+str(i[1]))


# # avg

# In[10]:


from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from base import Cache
from tqdm import tqdm
import pandas as pd 


# In[11]:


def get_embedding(f1_f2,f1):
    path = 'adv_userseq_'+f1_f2+'_word2vec.kv'
    wv = KeyedVectors.load(path, mmap='r')
    list_df = Cache.reload_cache('CACHE_list_df_adv_userseq_'+f1_f2+'.pkl')
    list_df.columns=['list',f1] 
    f = open('adv_userseq_'+f1_f2+'.txt','r')
    ind = 0
    buf = []
    for i in f:
        buf_ = np.zeros(64)
        for j in i.strip().split(' '):
            buf_ = buf_+wv[j]
        buf_ = buf_/len(i) # 求平均
        buf_f1 = list_df.at[ind, f1]
        buf__ = []
        buf_ = buf_.tolist()
        buf__.append(buf_)
        buf__.append(buf_f1)
        buf.append(buf__)
        ind = ind+1
    df_f1_list = pd.DataFrame(buf) 
    Cache.cache_data(df_f1_list, nm_marker='list_df_avg_adv_userseq_'+f1_f2)
    return 0


# In[12]:


f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],
              ['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],
              ['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],
              ['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],
              ['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],
              ['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],
              ['adv_prim_id','gender'],['adv_prim_id','residence']]
for i in tqdm(f1_f2_list):
    get_embedding(str(i[0])+'_'+str(i[1]),i[0])
    


# # adv_emb_mtx

# In[13]:


from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from base import Cache
from tqdm import tqdm
import pandas as pd 


# In[14]:


def get_embedding(f1_f2,f1):
    avg_f1 = Cache.reload_cache('CACHE_list_df_avg_adv_userseq_'+f1_f2+'.pkl')
    feature_tokens = avg_f1[[1]].values.flatten().astype(str).tolist()
    tokenizer = Tokenizer(num_words=len(feature_tokens)+1)
    tokenizer.fit_on_texts(feature_tokens)
    embedding_dim = 64
    embedding_matrix = np.random.randn(
        len(feature_tokens)+1, embedding_dim)
    avg_f1_copy = avg_f1.copy()
    avg_f1_copy = avg_f1_copy.set_index(1)
    
    for feature in feature_tokens:
        embedding_vector = np.array(avg_f1_copy.loc[int(feature),:].values[0])
        if embedding_vector is not None:
            index = tokenizer.texts_to_sequences([feature])[0][0]
            embedding_matrix[index] = embedding_vector
    return embedding_matrix


# In[15]:


if __name__ == '__main__': 
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],
                  ['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],
                  ['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],
                  ['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],
                  ['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],
                  ['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],
                  ['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    for i in tqdm(f1_f2_list):
        mtx = get_embedding(str(i[0])+'_'+str(i[1]),i[0])
        np.save(str(i[0])+'_'+str(i[1])+'_emb_mtx_adv_userseq_adv.npy',mtx)


# # user_emb_mtx

# In[16]:


from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tqdm import tqdm


# In[17]:


def get_embedding(path):
    path_kv = 'adv_userseq_'+path+'_word2vec.kv'
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


# In[18]:


f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],
              ['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],
              ['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],
              ['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],
              ['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],
              ['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],
              ['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
for i in tqdm(f1_f2_list):
    embedding_matrix = get_embedding(str(i[0])+'_'+str(i[1]))
    np.save(str(i[0])+'_'+str(i[1])+'_emb_mtx_adv_userseq_user.npy',embedding_matrix)


# # input_adv

# In[19]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from tqdm import tqdm
from base import Cache


# In[20]:


def input_w2v(f1_f2,all_data,f1):
    feature_seq = all_data[[f1]].values.flatten().astype(str).tolist()

    avg_f1 = Cache.reload_cache('CACHE_list_df_avg_adv_userseq_'+f1_f2+'.pkl')
    feature_tokens = avg_f1[[1]].values.flatten().astype(str).tolist()
    tokenizer = Tokenizer(num_words=len(feature_tokens)+1)
    tokenizer.fit_on_texts(feature_tokens)
    
    npy_path = f1_f2
    sequences = tokenizer.texts_to_sequences(feature_seq[:8672928])
    x_train = pad_sequences(
        sequences, maxlen=1, padding='post')
    print(x_train.shape)
    np.save(npy_path+'_adv_userseq_adv_train.npy', x_train)

    sequences = tokenizer.texts_to_sequences(feature_seq[8672928:])
    x_test = pad_sequences(
        sequences, maxlen=1, padding='post') 
    print(x_test.shape)
    np.save(npy_path+'_adv_userseq_adv_test.npy', x_test)


# In[21]:



f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],
              ['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],
              ['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],
              ['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],
              ['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],
              ['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],
              ['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
data = Cache.reload_cache('CACHE_data_sampling_pos1_neg5.pkl')
for i in tqdm(f1_f2_list):
    input_w2v(str(i[0])+'_'+str(i[1]),data,str(i[0]))


# # input_user

# In[22]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from tqdm import tqdm
from base import Cache


# In[23]:


def input_w2v(f1_f2,all_data,f2):
    feature_seq = all_data[[f2]].values.flatten().astype(str).tolist()

    path_kv = 'adv_userseq_'+f1_f2+'_word2vec.kv'
    wv = KeyedVectors.load(path_kv, mmap='r')
    feature_tokens = list(wv.vocab.keys())
    
    tokenizer = Tokenizer(num_words=len(feature_tokens)+1)
    tokenizer.fit_on_texts(feature_tokens)
    
    npy_path = f1_f2
    sequences = tokenizer.texts_to_sequences(feature_seq[:8672928])
    x_train = pad_sequences(
        sequences, maxlen=1, padding='post')
    print(x_train.shape)
    np.save(npy_path+'_adv_userseq_user_train.npy', x_train)

    sequences = tokenizer.texts_to_sequences(feature_seq[8672928:])
    x_test = pad_sequences(
        sequences, maxlen=1, padding='post') 
    print(x_test.shape)
    np.save(npy_path+'_adv_userseq_user_test.npy', x_test)


# In[24]:


f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],
              ['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],
              ['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],
              ['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],
              ['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],
              ['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],
              ['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
data = Cache.reload_cache('CACHE_data_sampling_pos1_neg5.pkl')
for i in tqdm(f1_f2_list):
    input_w2v(str(i[0])+'_'+str(i[1]),data,str(i[1]))
    


# In[ ]:




