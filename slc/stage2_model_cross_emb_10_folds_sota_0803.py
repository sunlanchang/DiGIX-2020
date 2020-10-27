#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)
from itertools import chain
from base import Cache
import gc
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from tqdm import tqdm
from base.trans_layer import Add, LayerNormalization
from base.trans_layer import MultiHeadAttention, PositionWiseFeedForward
from base.trans_layer import PositionEncoding
import joblib
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import optimizers, layers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, GlobalMaxPooling1D, Flatten
from tensorflow.keras.backend import concatenate
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.layers import CuDNNLSTM
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import  SparseFeat, DenseFeat, get_feature_names, build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input

import random
SEED = 1000
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
from multiprocessing import Pool
print(tf.__version__)

print('start!')
# data = Cache.reload_cache('CACHE_sampling_pro_feature.pkl')# 基础特征 0.797+
# # 给data 一个id列
# test_B = Cache.reload_cache('CACHE_test_B.pkl')
# test_B = test_B.reset_index()
# test_B.rename(columns={'index':'raw_index'},inplace=True)
# test_B['raw_index'] = test_B['raw_index']+41907133
# test_B = test_B[['raw_index','id']]
# data = data.merge(test_B,on='raw_index',how='left')
# data['id']=data['id'].fillna(-1)
# del test_B
# gc.collect()

# # 补充一些开源特征
# data.loc[data['pt_d']==9,'pt_d']= data.loc[data['pt_d']==9,'pt_d']-1
# # data['min_onlinerate'] = data['communication_onlinerate'].apply(lambda x:int(x.split('^')[0]))
# # data['max_onlinerate'] = data['communication_onlinerate'].apply(lambda x:int(x.split('^')[-1]))
# temp = data.groupby(['uid','pt_d']).agg({'label':'sum'}).reset_index()
# temp.columns = ['uid','pt_d','click_lastday_count']
# temp['pt_d']+=1
# data = data.merge(temp,on = ['uid','pt_d'],how = 'left')
# temp = data.groupby(['uid','pt_d']).agg({'adv_id':'count'}).reset_index()
# temp.columns = ['uid','pt_d','expo_lastday_counts']
# temp['pt_d']+=1
# data = data.merge(temp,on = ['uid','pt_d'],how = 'left')
# temp = data.groupby(['uid','pt_d']).agg({'adv_id':'count'}).reset_index()
# temp.columns = ['uid','pt_d','expo_day_counts']
# meancount = temp.loc[temp['pt_d']<8,'expo_day_counts'].mean()
# meancountalpha = meancount/ temp.loc[temp['pt_d']>=8,'expo_day_counts'].mean()
# temp.loc[temp['pt_d']>=8,'expo_day_counts'] = temp.loc[temp['pt_d']>=8,'expo_day_counts']*meancountalpha
# data = data.merge(temp,on = ['uid','pt_d'],how = 'left')
# for var in data.columns:
#     if data[var].isnull().sum()>0:
#         data[var] = data[var].fillna(data[var].mean())
#         print('fillna col:',var)

# Cache.cache_data(data,nm_marker='deepfm_data0929')

def reduce_mem(df, use_float16=False):
    start_mem = df.memory_usage().sum() / 1024**2
    tm_cols = df.select_dtypes('datetime').columns
    colsuse = [i for i in df.columns if i!= 'label']
    for col in colsuse:
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
# data = reduce_mem(data, use_float16=False)
data = Cache.reload_cache('CACHE_zlh_nn_feature_stage_2.pkl')
data = reduce_mem(data, use_float16=True)
# 重置index唯一值
del data['raw_index']
# del data['communication_onlinerate']
gc.collect()
data = data.reset_index(drop=True).reset_index()
# 加载cross emb
dense_feature_size=128
# m_user_0 = np.load('./cached_data/m0_user_stage2.npy').astype(np.float16)
m_user_1 = np.load('./cached_data/m1_user_stage2.npy').astype(np.float16)
# m_item_0 = np.load('./cached_data/m0_item_stage2.npy').astype(np.float16)
m_item_1 = np.load('./cached_data/m1_item_stage2.npy').astype(np.float16)
dataindex_base = np.load('./cached_data/dataindex_stage2.npy')
# 从matrix中取出采样后的输入
# m_user_0 = np.hstack([dataindex_base.reshape(-1,1),m_user_0])
# m_user_0 = m_user_0[list(data['index']),1:]
m_user_1 = np.hstack([dataindex_base.reshape(-1,1),m_user_1])
m_user_1 = m_user_1[list(data['index']),1:]
# m_item_0 = np.hstack([dataindex_base.reshape(-1,1),m_item_0])
# m_item_0 = m_item_0[list(data['index']),1:]
m_item_1 = np.hstack([dataindex_base.reshape(-1,1),m_item_1])
m_item_1 = m_item_1[list(data['index']),1:]
gc.collect()

# Cache.cache_data(data, nm_marker=f'zlh_nn_feature_stage_2')
# data = Cache.reload_cache('CACHE_zlh_nn_feature_stage_2.pkl')
# print(bug)
## window特征+2k


last_seq_list = ['tags','spread_app_id','task_id','adv_id','label']# 'inter_type_cd',

def get_emb_matrix(col):
    """
    inputs:    
    col 需要做成预训练emb_matrix的列
    
    加载：
    emb_dict 预训练的词向量
    word_emb_dict 字典
    id_list_dict 字典索引序列
    
    得出id_list_dict+emb_matrix
    """
    id_list_dict_all = Cache.reload_cache(f'CACHE_EMB_INPUTSEQ_stage2_{col}.pkl')
#     id_list_dict = id_list_dict_all['id_list']
#     key2index = id_list_dict_all['key2index']
#     emb = id_list_dict_all['emb']
    key_to_represent_rare = '-1'
    words = list(id_list_dict_all['emb'].keys())
    emb_size = id_list_dict_all['emb'][words[0]].shape[0]
    voc_size = len(words)
    emb_matrix = np.zeros((voc_size + 1, emb_size))
    # emb 中必须要有'-1' 作为index 0
    if '-1' not in id_list_dict_all['key2index'].keys():
        #  emb中无-1 为全词表数据！需要自行计算均值emb vec
        # 为embi 添加一个embedding
        # 这些词的vector求均值
        vector_low_frequency_words = np.zeros((emb_size,))
        for w in words:
            vector_low_frequency_words += id_list_dict_all['emb'][w]
        vector_low_frequency_words = vector_low_frequency_words / voc_size
        # emb添加一个key value
        id_list_dict_all['emb'][key_to_represent_rare] = vector_low_frequency_words
        # print(f'{col} file has no key_to_represent_rare add low frequency words and fill vector as:', vector_low_frequency_words)
    for k, idx in id_list_dict_all['key2index'].items():
        try:
            emb_matrix[idx, :] = id_list_dict_all['emb'][k]
        except KeyError:  # 如果k不在不在word_emb_dict中，则默认用max_key_to_represent_rare填充
            #                 print('find oov:',(k, idx))
            emb_matrix[idx, :] = id_list_dict_all['emb'][key_to_represent_rare]
    emb_matrix = np.float32(emb_matrix)
    return {col:[id_list_dict_all['id_list'],emb_matrix]}


with Pool(3) as p:
    res = p.map(get_emb_matrix, last_seq_list)
id_list_dict_emb_all = {}
for item in res:
    id_list_dict_emb_all.update(item)
del res,item
gc.collect()

GlobalSeqLength = 40
base_inputdim_dict = {}
for var in id_list_dict_emb_all.keys():
    base_inputdim_dict[var] = id_list_dict_emb_all[var][1].shape[0]
base_embdim_dict = {'inter_type_cd':32,'tags':32,'spread_app_id':32,'task_id':32,'adv_id':32,'label':32}# creat_type_cd
conv1d_info_dict = {'inter_type_cd':16,'tags':16,'spread_app_id':16,'task_id':16,'adv_id':32,'label':8}# creat_type_cd
TRAINABLE_DICT = {'inter_type_cd':False,'tags':False,'spread_app_id':False,'task_id':False,'adv_id':False,'label':False}# creat_type_cd
arr_name_list = list(id_list_dict_emb_all.keys())

def get_seq_input_layers(cols):
    '''
    seq 输入
    '''
    print("Prepare input layer:", cols)
    inputs_dict = {}
    for col in cols:
        inputs_dict[col] = tf.keras.Input(shape=(GlobalSeqLength, ),
                                          dtype="int32",
                                          name=col+'_seq_layer')
    return inputs_dict

def get_input_feature_layer(name=None,feature_shape=128,dtype="float16"):
    '''
    cross emb 直接输入
    '''
    input_layer = keras.Input(shape=(feature_shape,), dtype=dtype, name=name)
    return input_layer

def get_emb_layer(col, emb_matrix=None, seq_length=None, trainable=False):
    if seq_length is None:
        seq_length = GlobalSeqLength
    if trainable==True:
        emb_layer = tf.keras.layers.Embedding(base_inputdim_dict[col],
                                              base_embdim_dict[col],
                                              input_length=seq_length,
                                              dtype="float16",
                                              trainable=True)
    else:
        embedding_dim = emb_matrix.shape[-1]
        input_dim = emb_matrix.shape[0]
        emb_layer = tf.keras.layers.Embedding(input_dim,
                                              embedding_dim,
                                              input_length=seq_length,
                                              weights=[emb_matrix],
                                              dtype="float16",
                                              trainable=trainable)
    return emb_layer

def trans_net(inputs,masks ,hidden_unit=128):
    inputs = tf.keras.layers.Dropout(0.3)(inputs)
    encodings = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, padding='same', activation='relu')(inputs)
    # trans tunnel
    # pre Norm
    encodings = LayerNormalization()(encodings)
    # Masked-Multi-head-Attention
    masked_attention_out = MultiHeadAttention(8, encodings.shape[-1] // 8)([encodings, encodings, encodings, masks])
    # Add & Norm
    masked_attention_out = masked_attention_out+ encodings
    # Feed-Forward
    ff = PositionWiseFeedForward(encodings.shape[-1], hidden_unit)
    ff_out = ff(masked_attention_out)
    
    encodings = LayerNormalization()(ff_out)
    # Masked-Multi-head-Attention
    masked_attention_out = MultiHeadAttention(8, encodings.shape[-1] // 8)([encodings, encodings, encodings, masks])
    # Add & Norm
    masked_attention_out = masked_attention_out+ encodings
    # Feed-Forward
    ff = PositionWiseFeedForward(encodings.shape[-1], hidden_unit)
    ff_out = ff(masked_attention_out)
    
    # LSTM
    x = tf.keras.layers.Bidirectional(CuDNNLSTM(hidden_unit, return_sequences=True))(encodings)
    # linear
    x = tf.keras.layers.Conv1D(filters=encodings.shape[-1], kernel_size=1, padding='same', activation='relu')(x)
    # 3 项Add & Norm
    x = x + masked_attention_out + ff_out
    x = LayerNormalization()(x)
    return x

def cross_net(inputsi,inputj,hidden_unit=2):
    x = tf.keras.layers.concatenate([inputsi,inputj])
    x = tf.keras.layers.Dense(hidden_unit,activation='relu')(x)
    return x

def create_model(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary'):

    K.clear_session()
#!################################################################################################################
    inputs_all = [
#         get_input_feature_layer(name = 'user_0',feature_shape = dense_feature_size),
#                  get_input_feature_layer(name = 'item_0',feature_shape = dense_feature_size),
                 get_input_feature_layer(name = 'user_1',feature_shape = dense_feature_size),
                 get_input_feature_layer(name = 'item_1',feature_shape = dense_feature_size)
    ]
    # slotid_nettype
#     layer_user_0 = inputs_all[0]
#     layer_user_0 = K.expand_dims(layer_user_0, 1)
#     layer_item_0 = inputs_all[1]
#     layer_item_0 = K.expand_dims(layer_item_0, 1)
    layer_user_1 = inputs_all[0]
    layer_user_1 = K.expand_dims(layer_user_1, 1)
    layer_item_1 = inputs_all[1]
    layer_item_1 = K.expand_dims(layer_item_1, 1)
#     cross_emb_out0 = cross_net(layer_user_0,layer_item_0)
    cross_emb_out1 = cross_net(layer_user_1,layer_item_1)
#     cross_emb_out = tf.keras.layers.concatenate([cross_emb_out0,cross_emb_out1])
    cross_emb_out = tf.squeeze(cross_emb_out1,[1])
#!################################################################################################################
    seq_inputs_dict = get_seq_input_layers(cols=arr_name_list)
    inputs_all = inputs_all + list(seq_inputs_dict.values())  # 输入层list
    masks = tf.equal(seq_inputs_dict['task_id'], 0)
    # 普通序列+label序列
    layers2concat = []
    for index, col in enumerate(arr_name_list):
        print(col, 'get embedding!')
        emb_layer = get_emb_layer(col, trainable=TRAINABLE_DICT[col],emb_matrix=id_list_dict_emb_all[col][1])
        x = emb_layer(seq_inputs_dict[col])
        if conv1d_info_dict[col] > -1:
            cov_layer = tf.keras.layers.Conv1D(filters=conv1d_info_dict[col],
                                               kernel_size=1,
                                               activation='relu')
            x = cov_layer(x)
        layers2concat.append(x)
    x = keras.layers.concatenate(layers2concat)
#!################################################################################################################
#!mix1
    x = trans_net(x, masks, hidden_unit=256)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()
    average_pool = tf.keras.layers.GlobalAveragePooling1D()
    xmaxpool = max_pool(x)
    xmeanpool = average_pool(x)

    trans_output = tf.keras.layers.concatenate([xmaxpool, xmeanpool])


#!################################################################################################################
#!mix2
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    mix = concatenate([cross_emb_out,trans_output, dnn_input], axis=-1)  # !#mix

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(mix)

    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

#!################################################################################################################

    model = Model(inputs=inputs_all+[features],
                  outputs=[output])
    print(model.summary())
    return model

def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    Usage:
     model.compile(loss=[multi_category_focal_loss2(
         alpha=0.35, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha +             (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed

  # 输入列
sparse_features=['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id',
                 'tags', 'app_first_class', 'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size',
                 'career', 'gender', 'net_type', 'residence', 'his_app_size', 'his_on_shelf_time', 'app_score', 'emui_dev',
                 'list_time', 'device_price', 'up_life_duration', 'up_membership_grade', 'membership_life_duration', 
                 'consume_purchase', 'communication_avgonline_30d', 'indu_name', 
                 'communication_onlinerate_1',
 'communication_onlinerate_2',
 'communication_onlinerate_3',
 'communication_onlinerate_4',
 'communication_onlinerate_5',
 'communication_onlinerate_6',
 'communication_onlinerate_7',
 'communication_onlinerate_8',
 'communication_onlinerate_9',
 'communication_onlinerate_10',
 'communication_onlinerate_11',
 'communication_onlinerate_12',
 'communication_onlinerate_13',
 'communication_onlinerate_14',
 'communication_onlinerate_15',
 'communication_onlinerate_16',
 'communication_onlinerate_17',
 'communication_onlinerate_18',
 'communication_onlinerate_19',
 'communication_onlinerate_20',
 'communication_onlinerate_21',
 'communication_onlinerate_22',
 'communication_onlinerate_23']
set1 = data.query('pt_d<8').copy()
set2 = data.query('pt_d>=8').copy()
print('processing unknow')
for var in tqdm(sparse_features):
    if var.find('cmr')==-1:
        setnotknow = set(set1[var]).intersection(set(set2[var]))
        if len(setnotknow)>0:
            data.loc[~data[var].isin(setnotknow),var] = -1
del set1,set2
gc.collect()

dense_features=[i for i in data.columns if i not in sparse_features+['index','id','uid','level_0','pt_d','label','raw_index']]

droplist = []
for var in tqdm(sparse_features+dense_features):
    if data.query('pt_d>=8')[var].nunique()<2 or data.query('pt_d>=8')[var].count()<2:
        droplist.append(var)
for var in droplist:
    if var in dense_features:
        dense_features.remove(var)
    if var in sparse_features:
        sparse_features.remove(var)
    if var in data.columns:
        print(f'we drop col:{var}')
        del data[var]
gc.collect()
# dense_features = dense_features+add_fe
print('sparse_features:')
print(sparse_features)
print('dense_features:')
print(dense_features)

# 特征处理
# Label Encoding for sparse features,and do simple Transformation for dense features
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat]).astype(np.int32)
for feat in tqdm(dense_features):
    mms = MinMaxScaler(feature_range=(0, 1))
    data[feat] = mms.fit_transform(data[feat].values.reshape(-1,1)).astype(np.float16)
del mms,lbe
# print('find droplist:',droplist)
gc.collect()
#!################################################################################################################
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=8)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print('feature_names finish!')

# callback
lr_list = [0.001, 0.001, 0.001, 0.0005, 0.00025, 0.000125, 6.25e-05, 3.125e-05, 2e-05, 2e-05, 2e-05]
def scheduler(epoch):
    if epoch < len(lr_list):
        return lr_list[epoch]
    else:
        return 2.5e-6
def get_callbacks(fold, if_valid=True):
    '''
    :param count:
    :return:
    '''
    checkpoint_dir = 'models'
    checkpoint_prefix = os.path.join(
        checkpoint_dir, f"ckpt_zlhnn_model0929_{fold}_fold_{if_valid}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                             save_weights_only=True,
                                                             monitor='val_AUC',
                                                             verbose=1,
                                                             save_freq="epoch",
                                                             save_best_only=True,
                                                             mode='max')
    reduce_lr_callback_trainall = tf.keras.callbacks.LearningRateScheduler(
        scheduler)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_AUC",
        min_delta=0.00001,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
    )
    csv_log_callback = tf.keras.callbacks.CSVLogger(
        filename=f'./logs/model_zlhnn_model0929_{fold}_.log', separator=",", append=True)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_AUC',
                                                              factor=0.25,
                                                              patience=1,
                                                              min_delta=2e-4,
                                                              min_lr=2e-5)
    if if_valid:
        callbacks = [checkpoint_callback, csv_log_callback,
                     earlystop_callback, reduce_lr_callback]
    else:
        callbacks = [checkpoint_callback,
                     csv_log_callback,  reduce_lr_callback_trainall]
    return callbacks

# compile model
# del model
# gc.collect()
bs = 2048*2
count = 0
random_state = 2
set1 = data.query('pt_d<8').copy()
set2 = data.query('pt_d>=8').copy()
del data
gc.collect()
idtrain=list(set1['index'])
idtest=list(set2['index'])
res = set2[['id','index']]
res['probability']=0
trainvalid = set1[['id','index']]
trainvalid['probability']=0
skf = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)# 抽90% 训练
for i, (train_index, test_index) in enumerate(skf.split(set1, set1['label'])):
    print("FOLD | ", count+1)
    print("###"*35)
    gc.collect()
    print(' model compile start ……')
    try:
        del model
        gc.collect()
        K.clear_session()
    except:
        pass
    model = create_model(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                         dnn_hidden_units=(512, 256, 128),dnn_dropout=0.0,task='binary')
    model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=5e-4), loss=multi_category_focal_loss2(alpha=0.35),
    metrics=['AUC'])
    print(' model compile finish ……')
    # 模型输入
    # 训练集
    online_train_model_input = {}
#     online_train_model_input['user_0']=m_user_0[idtrain][train_index]
#     online_train_model_input['item_0']=m_item_0[idtrain][train_index]
    online_train_model_input['user_1']=m_user_1[idtrain][train_index]
    online_train_model_input['item_1']=m_item_1[idtrain][train_index]
    for var in id_list_dict_emb_all.keys():
        online_train_model_input[var+'_seq_layer']= id_list_dict_emb_all[var][0][idtrain][train_index]
    online_train_model_input.update({name: set1[name].values[train_index] for name in tqdm(feature_names)})
    y_true_train = set1['label'].values[train_index]
    print('train input built!')
    # 验证集
    online_valid_model_input = {}
#     online_valid_model_input['user_0']=m_user_0[idtrain][test_index]
#     online_valid_model_input['item_0']=m_item_0[idtrain][test_index]
    online_valid_model_input['user_1']=m_user_1[idtrain][test_index]
    online_valid_model_input['item_1']=m_item_1[idtrain][test_index]
    for var in id_list_dict_emb_all.keys():
        online_valid_model_input[var+'_seq_layer']= id_list_dict_emb_all[var][0][idtrain][test_index]
    online_valid_model_input.update({name: set1[name].values[test_index] for name in tqdm(feature_names)})
    y_true_valid = set1['label'].values[test_index]
    print('valid input built!')

    callbacks = get_callbacks(i)
    hist = model.fit(online_train_model_input,y_true_train,
        epochs=25, 
        batch_size=bs,
        verbose=1, 
        callbacks=callbacks,
        validation_data=(online_valid_model_input, y_true_valid))
    print(hist.history)
    del online_train_model_input
    gc.collect()
    # 测试集
    online_test_model_input = {}
#     online_test_model_input['user_0']=m_user_0[idtest]
#     online_test_model_input['item_0']=m_item_0[idtest]
    online_test_model_input['user_1']=m_user_1[idtest]
    online_test_model_input['item_1']=m_item_1[idtest]
    for var in id_list_dict_emb_all.keys():
        online_test_model_input[var+'_seq_layer']= id_list_dict_emb_all[var][0][idtest]
    online_test_model_input.update({name: set2[name].values for name in tqdm(feature_names)})
    print('test input built!')
    # 预测
    y_pre = model.predict(online_test_model_input, verbose=2, batch_size=1024)
    np.save(f'./cached_data/model0929_test_f{i}.npy',y_pre)
    res['probability'] = res['probability'].values.reshape(-1,1)+y_pre.reshape(-1,1)/10
    # 预测验证集
    y_valid = model.predict(online_valid_model_input, verbose=2, batch_size=1024)
    np.save(f'./cached_data/model0929_valid_f{i}.npy',y_valid)
    trainvalid['probability'].iloc[test_index]= y_valid.ravel()
    del online_valid_model_input,online_test_model_input
    gc.collect()
    count+=1
# 整体保存
df_save = pd.concat([trainvalid,res],axis=0)
df_save.to_pickle('model0929_10_folds_oof.pkl')
# subs
res = res[['id','probability']]
res = res.sort_values('id')
res.to_csv('./subs/submission_nn_model0929_10_foldsTIONE_0.csv', index=False)


# In[ ]:




