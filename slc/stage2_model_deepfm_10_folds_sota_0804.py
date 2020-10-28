#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pdb
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from multiprocessing import Pool
import random
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.layers.interaction import FM
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras.layers import CuDNNLSTM
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, GlobalMaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, layers, losses
from tensorflow import keras
import tensorflow as tf
from base.trans_layer import PositionEncoding
from base.trans_layer import MultiHeadAttention, PositionWiseFeedForward
from base.trans_layer import Add, LayerNormalization
import numpy as np
import pandas as pd
from itertools import chain
import gc
from base import Cache

import os
from tqdm import tqdm
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# from tensorflow.keras.utils import multi_gpu_model


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# slc 显卡的个数需要修改
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
gpus = tf.config.experimental.list_physical_devices('GPU')
for device in gpus:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.experimental.set_memory_growth(gpus[0], True)


# ## 加载基础特征 44维sparse特征 249维dense特征 总计293维

# slc
deepfm_data = Cache.reload_cache('CACHE_sampling_pro_feature.pkl')
# label = deepfm_data[deepfm_data["raw_index"]<6000_0000]['label']# .values
# slc 训练集个数修正为6千万
label = deepfm_data[deepfm_data["raw_index"] < 6000_0000]['label']  # .values


# ## 序列特征 预训练embedding加载 总计做了6个广告,label序列的过去40个记录序列


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
    id_list_dict_all = Cache.reload_cache(
        f'CACHE_EMB_INPUTSEQ_stage2_{col}.pkl')
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
    return {col: [id_list_dict_all['id_list'], emb_matrix]}


# slc
# last_seq_list = ['label', 'creat_type_cd', 'tags','spread_app_id', 'task_id', 'adv_id']
# last_seq_list = []
# with Pool(6) as p:
    # res = p.map(get_emb_matrix, last_seq_list)
last_seq_list = ['label']
res = get_emb_matrix('label')
id_list_dict_emb_all = {}
for item in res:
    id_list_dict_emb_all.update(item)
# del res, item
del res
gc.collect()

GlobalSeqLength = 40
base_inputdim_dict = {}
for var in id_list_dict_emb_all.keys():
    base_inputdim_dict[var] = id_list_dict_emb_all[var][1].shape[0]
# base_embdim_dict = {'creat_type_cd': 32, 'tags': 32,
#                     'spread_app_id': 32, 'task_id': 32, 'adv_id': 32, 'label': 32}
# conv1d_info_dict = {'creat_type_cd': 8, 'tags': 8,
#                     'spread_app_id': 8, 'task_id': 16, 'adv_id': 32, 'label': 8}
# TRAINABLE_DICT = {'creat_type_cd': False, 'tags': False,
#                   'spread_app_id': False, 'task_id': False, 'adv_id': False, 'label': False}
base_embdim_dict = {'label': 32}
conv1d_info_dict = {'label': 8}
TRAINABLE_DICT = {'label': False}
# base_embdim_dict = {}
# conv1d_info_dict = {}
# TRAINABLE_DICT = {}
arr_name_list = list(id_list_dict_emb_all.keys())  # 过去行为序列

# ## 神经网络部分 transformer学习序列特征，user_item交叉结构学习相似度，deepfm学习基础特征，使用focalloss做二分类


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

        alpha_t = y_true * alpha + \
            (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


def trans_net(inputs, masks, hidden_unit=128):
    inputs = tf.keras.layers.Dropout(0.3)(inputs)
    encodings = tf.keras.layers.Conv1D(
        filters=inputs.shape[-1], kernel_size=1, padding='same', activation='relu')(inputs)
    # trans tunnel
    for i in range(1):
        # pre Norm
        encodings = LayerNormalization()(encodings)
        # Masked-Multi-head-Attention
        masked_attention_out = MultiHeadAttention(
            8, encodings.shape[-1] // 8)([encodings, encodings, encodings, masks])
        # Add & Norm
        masked_attention_out = masked_attention_out + encodings
        # Feed-Forward
        ff = PositionWiseFeedForward(encodings.shape[-1], hidden_unit)
        ff_out = ff(masked_attention_out)
    # LSTM
    x = tf.keras.layers.Bidirectional(
        CuDNNLSTM(hidden_unit, return_sequences=True))(encodings)
    # linear
    x = tf.keras.layers.Conv1D(
        filters=encodings.shape[-1], kernel_size=1, padding='same', activation='relu')(x)
    # 3 项Add & Norm
    x = x + masked_attention_out + ff_out
    x = LayerNormalization()(x)
    return x


def get_seq_input_layers(cols):
    print("Prepare input layer:", cols)
    inputs_dict = {}
    for col in cols:
        inputs_dict[col] = tf.keras.Input(shape=(GlobalSeqLength, ),
                                          dtype="int32",
                                          name=col+'_seq_layer')
    return inputs_dict


def get_emb_layer(col, emb_matrix=None, seq_length=None, trainable=False):
    if seq_length is None:
        seq_length = GlobalSeqLength
    if trainable == True:
        emb_layer = tf.keras.layers.Embedding(base_inputdim_dict[col],
                                              base_embdim_dict[col],
                                              input_length=seq_length,
                                              dtype="float32",
                                              trainable=True)
    else:
        embedding_dim = emb_matrix.shape[-1]
        input_dim = emb_matrix.shape[0]
        emb_layer = tf.keras.layers.Embedding(input_dim,
                                              embedding_dim,
                                              input_length=seq_length,
                                              weights=[emb_matrix],
                                              dtype="float32",
                                              trainable=trainable)
    return emb_layer


def M(emb_mtx_f1_1, emb_mtx_f1_2, emb_mtx_f1_3, emb_mtx_f1_4, emb_mtx_f1_5,
      emb_mtx_f1_6, emb_mtx_f1_7, emb_mtx_f1_8, emb_mtx_f1_9, emb_mtx_f1_10,
      emb_mtx_f1_11, emb_mtx_f1_12, emb_mtx_f1_13, emb_mtx_f1_14, emb_mtx_f1_15,
      emb_mtx_f1_16, emb_mtx_f1_17, emb_mtx_f1_18, emb_mtx_f1_19, emb_mtx_f1_20,
      emb_mtx_f1_21, emb_mtx_f1_22, emb_mtx_f1_23, emb_mtx_f1_24, emb_mtx_f1_25,
      emb_mtx_f1_26, emb_mtx_f1_27, emb_mtx_f1_28, emb_mtx_f1_29, emb_mtx_f1_30,
      emb_mtx_f1_31, emb_mtx_f1_32, emb_mtx_f1_33, emb_mtx_f1_34, emb_mtx_f1_35,
      #!###
      emb_mtx_f2_1, emb_mtx_f2_2, emb_mtx_f2_3, emb_mtx_f2_4, emb_mtx_f2_5,
      emb_mtx_f2_6, emb_mtx_f2_7, emb_mtx_f2_8, emb_mtx_f2_9, emb_mtx_f2_10,
      emb_mtx_f2_11, emb_mtx_f2_12, emb_mtx_f2_13, emb_mtx_f2_14, emb_mtx_f2_15,
      emb_mtx_f2_16, emb_mtx_f2_17, emb_mtx_f2_18, emb_mtx_f2_19, emb_mtx_f2_20,
      emb_mtx_f2_21, emb_mtx_f2_22, emb_mtx_f2_23, emb_mtx_f2_24, emb_mtx_f2_25,
      emb_mtx_f2_26, emb_mtx_f2_27, emb_mtx_f2_28, emb_mtx_f2_29, emb_mtx_f2_30,
      emb_mtx_f2_31, emb_mtx_f2_32, emb_mtx_f2_33, emb_mtx_f2_34, emb_mtx_f2_35,
      #!###
      linear_feature_columns, dnn_feature_columns, fm_group=[
          DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
      l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
      dnn_activation='elu', dnn_use_bn=False, task='binary'):

    K.clear_session()

#!################################################################################################################
    input_1_f1 = Input(shape=(1,), name='input_1_f1_layer')
    input_1_f2 = Input(shape=(1,), name='input_1_f2_layer')

    x1_1 = Embedding(input_dim=emb_mtx_f1_1.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_1],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_1_f1)
    x1_1 = Flatten()(x1_1)

    x2_1 = Embedding(input_dim=emb_mtx_f2_1.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_1],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_1_f2)
    x2_1 = Flatten()(x2_1)

    concat_1 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_1 = Dense(10, activation='elu',)(concat_1)
#!################################################################################################################
    input_2_f1 = Input(shape=(1,), name='input_2_f1_layer')
    input_2_f2 = Input(shape=(1,), name='input_2_f2_layer')

    x1_2 = Embedding(input_dim=emb_mtx_f1_2.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_2],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_2_f1)
    x1_2 = Flatten()(x1_2)

    x2_2 = Embedding(input_dim=emb_mtx_f2_2.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_2],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_2_f2)
    x2_2 = Flatten()(x2_2)

    concat_2 = concatenate([x1_2, x2_2], axis=-1)

    output_f1_f2_2 = Dense(10, activation='elu',)(concat_2)
#!################################################################################################################
    input_3_f1 = Input(shape=(1,), name='input_3_f1_layer')
    input_3_f2 = Input(shape=(1,), name='input_3_f2_layer')

    x1_3 = Embedding(input_dim=emb_mtx_f1_3.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_3],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_3_f1)
    x1_3 = Flatten()(x1_3)

    x2_3 = Embedding(input_dim=emb_mtx_f2_3.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_3],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_3_f2)
    x2_3 = Flatten()(x2_3)

    concat_3 = concatenate([x1_3, x2_3], axis=-1)

    output_f1_f2_3 = Dense(10, activation='elu',)(concat_3)
#!################################################################################################################
    input_4_f1 = Input(shape=(1,), name='input_4_f1_layer')
    input_4_f2 = Input(shape=(1,), name='input_4_f2_layer')

    x1_4 = Embedding(input_dim=emb_mtx_f1_4.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_4],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_4_f1)
    x1_4 = Flatten()(x1_4)

    x2_4 = Embedding(input_dim=emb_mtx_f2_4.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_4],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_4_f2)
    x2_4 = Flatten()(x2_4)

    concat_4 = concatenate([x1_4, x2_4], axis=-1)

    output_f1_f2_4 = Dense(10, activation='elu',)(concat_4)
#!################################################################################################################
    input_5_f1 = Input(shape=(1,), name='input_5_f1_layer')
    input_5_f2 = Input(shape=(1,), name='input_5_f2_layer')

    x1_5 = Embedding(input_dim=emb_mtx_f1_5.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_5],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_5_f1)
    x1_5 = Flatten()(x1_5)

    x2_5 = Embedding(input_dim=emb_mtx_f2_5.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_5],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_5_f2)
    x2_5 = Flatten()(x2_5)

    concat_5 = concatenate([x1_5, x2_5], axis=-1)

    output_f1_f2_5 = Dense(10, activation='elu',)(concat_5)
#!################################################################################################################
    input_6_f1 = Input(shape=(1,), name='input_6_f1_layer')
    input_6_f2 = Input(shape=(1,), name='input_6_f2_layer')

    x1_6 = Embedding(input_dim=emb_mtx_f1_6.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_6],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_6_f1)
    x1_6 = Flatten()(x1_6)

    x2_6 = Embedding(input_dim=emb_mtx_f2_6.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_6],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_6_f2)
    x2_6 = Flatten()(x2_6)

    concat_6 = concatenate([x1_6, x2_6], axis=-1)

    output_f1_f2_6 = Dense(10, activation='elu',)(concat_6)
#!################################################################################################################
    input_7_f1 = Input(shape=(1,), name='input_7_f1_layer')
    input_7_f2 = Input(shape=(1,), name='input_7_f2_layer')

    x1_7 = Embedding(input_dim=emb_mtx_f1_7.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_7],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_7_f1)
    x1_7 = Flatten()(x1_7)

    x2_7 = Embedding(input_dim=emb_mtx_f2_7.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_7],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_7_f2)
    x2_7 = Flatten()(x2_7)

    concat_7 = concatenate([x1_7, x2_7], axis=-1)

    output_f1_f2_7 = Dense(10, activation='elu',)(concat_7)
#!################################################################################################################
    input_8_f1 = Input(shape=(1,), name='input_8_f1_layer')
    input_8_f2 = Input(shape=(1,), name='input_8_f2_layer')

    x1_8 = Embedding(input_dim=emb_mtx_f1_8.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_8],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_8_f1)
    x1_8 = Flatten()(x1_8)

    x2_8 = Embedding(input_dim=emb_mtx_f2_8.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_8],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_8_f2)
    x2_8 = Flatten()(x2_8)

    concat_8 = concatenate([x1_8, x2_8], axis=-1)

    output_f1_f2_8 = Dense(10, activation='elu',)(concat_8)
#!################################################################################################################
    input_9_f1 = Input(shape=(1,), name='input_9_f1_layer')
    input_9_f2 = Input(shape=(1,), name='input_9_f2_layer')

    x1_9 = Embedding(input_dim=emb_mtx_f1_9.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f1_9],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_9_f1)
    x1_9 = Flatten()(x1_9)

    x2_9 = Embedding(input_dim=emb_mtx_f2_9.shape[0],
                     output_dim=64,
                     weights=[emb_mtx_f2_9],
                     trainable=False,
                     input_length=1,
                     mask_zero=True)(input_9_f2)
    x2_9 = Flatten()(x2_9)

    concat_9 = concatenate([x1_9, x2_9], axis=-1)

    output_f1_f2_9 = Dense(10, activation='elu',)(concat_9)
#!################################################################################################################
    input_10_f1 = Input(shape=(1,), name='input_10_f1_layer')
    input_10_f2 = Input(shape=(1,), name='input_10_f2_layer')

    x1_10 = Embedding(input_dim=emb_mtx_f1_10.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_10],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_10_f1)
    x1_10 = Flatten()(x1_10)

    x2_10 = Embedding(input_dim=emb_mtx_f2_10.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_10],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_10_f2)
    x2_10 = Flatten()(x2_10)

    concat_10 = concatenate([x1_10, x2_10], axis=-1)

    output_f1_f2_10 = Dense(10, activation='elu',)(concat_10)
#!################################################################################################################
    input_11_f1 = Input(shape=(1,), name='input_11_f1_layer')
    input_11_f2 = Input(shape=(1,), name='input_11_f2_layer')

    x1_11 = Embedding(input_dim=emb_mtx_f1_11.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_11],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_11_f1)
    x1_11 = Flatten()(x1_11)

    x2_11 = Embedding(input_dim=emb_mtx_f2_11.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_11],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_11_f2)
    x2_11 = Flatten()(x2_11)

    concat_11 = concatenate([x1_11, x2_11], axis=-1)

    output_f1_f2_11 = Dense(10, activation='elu',)(concat_11)
#!################################################################################################################
    input_12_f1 = Input(shape=(1,), name='input_12_f1_layer')
    input_12_f2 = Input(shape=(1,), name='input_12_f2_layer')

    x1_12 = Embedding(input_dim=emb_mtx_f1_12.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_12],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_12_f1)
    x1_12 = Flatten()(x1_12)

    x2_12 = Embedding(input_dim=emb_mtx_f2_12.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_12],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_12_f2)
    x2_12 = Flatten()(x2_12)

    concat_12 = concatenate([x1_12, x2_12], axis=-1)

    output_f1_f2_12 = Dense(10, activation='elu',)(concat_12)
#!################################################################################################################
    input_13_f1 = Input(shape=(1,), name='input_13_f1_layer')
    input_13_f2 = Input(shape=(1,), name='input_13_f2_layer')

    x1_13 = Embedding(input_dim=emb_mtx_f1_13.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_13],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_13_f1)
    x1_13 = Flatten()(x1_13)

    x2_13 = Embedding(input_dim=emb_mtx_f2_13.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_13],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_13_f2)
    x2_13 = Flatten()(x2_13)

    concat_13 = concatenate([x1_13, x2_13], axis=-1)

    output_f1_f2_13 = Dense(10, activation='elu',)(concat_13)
#!################################################################################################################
    input_14_f1 = Input(shape=(1,), name='input_14_f1_layer')
    input_14_f2 = Input(shape=(1,), name='input_14_f2_layer')

    x1_14 = Embedding(input_dim=emb_mtx_f1_14.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_14],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_14_f1)
    x1_14 = Flatten()(x1_14)

    x2_14 = Embedding(input_dim=emb_mtx_f2_14.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_14],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_14_f2)
    x2_14 = Flatten()(x2_14)

    concat_14 = concatenate([x1_14, x2_14], axis=-1)

    output_f1_f2_14 = Dense(10, activation='elu',)(concat_14)
#!################################################################################################################
    input_15_f1 = Input(shape=(1,), name='input_15_f1_layer')
    input_15_f2 = Input(shape=(1,), name='input_15_f2_layer')

    x1_15 = Embedding(input_dim=emb_mtx_f1_15.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_15],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_15_f1)
    x1_15 = Flatten()(x1_15)

    x2_15 = Embedding(input_dim=emb_mtx_f2_15.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_15],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_15_f2)
    x2_15 = Flatten()(x2_15)

    concat_15 = concatenate([x1_15, x2_15], axis=-1)

    output_f1_f2_15 = Dense(10, activation='elu',)(concat_15)
#!################################################################################################################
    input_16_f1 = Input(shape=(1,), name='input_16_f1_layer')
    input_16_f2 = Input(shape=(1,), name='input_16_f2_layer')

    x1_16 = Embedding(input_dim=emb_mtx_f1_16.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_16],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_16_f1)
    x1_16 = Flatten()(x1_16)

    x2_16 = Embedding(input_dim=emb_mtx_f2_16.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_16],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_16_f2)
    x2_16 = Flatten()(x2_16)

    concat_16 = concatenate([x1_16, x2_16], axis=-1)

    output_f1_f2_16 = Dense(10, activation='elu',)(concat_16)
#!################################################################################################################
    input_17_f1 = Input(shape=(1,), name='input_17_f1_layer')
    input_17_f2 = Input(shape=(1,), name='input_17_f2_layer')

    x1_17 = Embedding(input_dim=emb_mtx_f1_17.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_17],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_17_f1)
    x1_17 = Flatten()(x1_17)

    x2_17 = Embedding(input_dim=emb_mtx_f2_17.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_17],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_17_f2)
    x2_17 = Flatten()(x2_17)

    concat_17 = concatenate([x1_17, x2_17], axis=-1)

    output_f1_f2_17 = Dense(10, activation='elu',)(concat_17)
#!################################################################################################################
    input_18_f1 = Input(shape=(1,), name='input_18_f1_layer')
    input_18_f2 = Input(shape=(1,), name='input_18_f2_layer')

    x1_18 = Embedding(input_dim=emb_mtx_f1_18.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_18],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_18_f1)
    x1_18 = Flatten()(x1_18)

    x2_18 = Embedding(input_dim=emb_mtx_f2_18.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_18],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_18_f2)
    x2_18 = Flatten()(x2_18)

    concat_18 = concatenate([x1_18, x2_18], axis=-1)

    output_f1_f2_18 = Dense(10, activation='elu',)(concat_18)
#!################################################################################################################
    input_19_f1 = Input(shape=(1,), name='input_19_f1_layer')
    input_19_f2 = Input(shape=(1,), name='input_19_f2_layer')

    x1_19 = Embedding(input_dim=emb_mtx_f1_19.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_19],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_19_f1)
    x1_19 = Flatten()(x1_19)

    x2_19 = Embedding(input_dim=emb_mtx_f2_19.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_19],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_19_f2)
    x2_19 = Flatten()(x2_19)

    concat_19 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_19 = Dense(10, activation='elu',)(concat_19)
#!################################################################################################################
    input_20_f1 = Input(shape=(1,), name='input_20_f1_layer')
    input_20_f2 = Input(shape=(1,), name='input_20_f2_layer')

    x1_20 = Embedding(input_dim=emb_mtx_f1_20.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_20],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_20_f1)
    x1_20 = Flatten()(x1_20)

    x2_20 = Embedding(input_dim=emb_mtx_f2_20.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_20],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_20_f2)
    x2_20 = Flatten()(x2_20)

    concat_20 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_20 = Dense(10, activation='elu',)(concat_20)
#!################################################################################################################
    input_21_f1 = Input(shape=(1,), name='input_21_f1_layer')
    input_21_f2 = Input(shape=(1,), name='input_21_f2_layer')

    x1_21 = Embedding(input_dim=emb_mtx_f1_21.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_21],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_21_f1)
    x1_21 = Flatten()(x1_21)

    x2_21 = Embedding(input_dim=emb_mtx_f2_21.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_21],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_21_f2)
    x2_21 = Flatten()(x2_21)

    concat_21 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_21 = Dense(10, activation='elu',)(concat_21)
#!################################################################################################################
    input_22_f1 = Input(shape=(1,), name='input_22_f1_layer')
    input_22_f2 = Input(shape=(1,), name='input_22_f2_layer')

    x1_22 = Embedding(input_dim=emb_mtx_f1_22.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_22],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_22_f1)
    x1_22 = Flatten()(x1_22)

    x2_22 = Embedding(input_dim=emb_mtx_f2_22.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_22],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_22_f2)
    x2_22 = Flatten()(x2_22)

    concat_22 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_22 = Dense(10, activation='elu',)(concat_22)
#!################################################################################################################
    input_23_f1 = Input(shape=(1,), name='input_23_f1_layer')
    input_23_f2 = Input(shape=(1,), name='input_23_f2_layer')

    x1_23 = Embedding(input_dim=emb_mtx_f1_23.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_23],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_23_f1)
    x1_23 = Flatten()(x1_23)

    x2_23 = Embedding(input_dim=emb_mtx_f2_23.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_23],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_23_f2)
    x2_23 = Flatten()(x2_23)

    concat_23 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_23 = Dense(10, activation='elu',)(concat_23)
#!################################################################################################################
    input_24_f1 = Input(shape=(1,), name='input_24_f1_layer')
    input_24_f2 = Input(shape=(1,), name='input_24_f2_layer')

    x1_24 = Embedding(input_dim=emb_mtx_f1_24.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_24],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_24_f1)
    x1_24 = Flatten()(x1_24)

    x2_24 = Embedding(input_dim=emb_mtx_f2_24.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_24],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_24_f2)
    x2_24 = Flatten()(x2_24)

    concat_24 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_24 = Dense(10, activation='elu',)(concat_24)
#!################################################################################################################
    input_25_f1 = Input(shape=(1,), name='input_25_f1_layer')
    input_25_f2 = Input(shape=(1,), name='input_25_f2_layer')

    x1_25 = Embedding(input_dim=emb_mtx_f1_25.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_25],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_25_f1)
    x1_25 = Flatten()(x1_25)

    x2_25 = Embedding(input_dim=emb_mtx_f2_25.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_25],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_25_f2)
    x2_25 = Flatten()(x2_25)

    concat_25 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_25 = Dense(10, activation='elu',)(concat_25)
#!################################################################################################################
    input_26_f1 = Input(shape=(1,), name='input_26_f1_layer')
    input_26_f2 = Input(shape=(1,), name='input_26_f2_layer')

    x1_26 = Embedding(input_dim=emb_mtx_f1_26.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_26],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_26_f1)
    x1_26 = Flatten()(x1_26)

    x2_26 = Embedding(input_dim=emb_mtx_f2_26.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_26],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_26_f2)
    x2_26 = Flatten()(x2_26)

    concat_26 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_26 = Dense(10, activation='elu',)(concat_26)

#!################################################################################################################
    input_27_f1 = Input(shape=(1,), name='input_27_f1_layer')
    input_27_f2 = Input(shape=(1,), name='input_27_f2_layer')

    x1_27 = Embedding(input_dim=emb_mtx_f1_27.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_27],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_27_f1)
    x1_27 = Flatten()(x1_27)

    x2_27 = Embedding(input_dim=emb_mtx_f2_27.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_27],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_27_f2)
    x2_27 = Flatten()(x2_27)

    concat_27 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_27 = Dense(10, activation='elu',)(concat_27)
#!################################################################################################################
    input_28_f1 = Input(shape=(1,), name='input_28_f1_layer')
    input_28_f2 = Input(shape=(1,), name='input_28_f2_layer')

    x1_28 = Embedding(input_dim=emb_mtx_f1_28.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_28],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_28_f1)
    x1_28 = Flatten()(x1_28)

    x2_28 = Embedding(input_dim=emb_mtx_f2_28.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_28],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_28_f2)
    x2_28 = Flatten()(x2_28)

    concat_28 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_28 = Dense(10, activation='elu',)(concat_28)
#!################################################################################################################
    input_29_f1 = Input(shape=(1,), name='input_29_f1_layer')
    input_29_f2 = Input(shape=(1,), name='input_29_f2_layer')

    x1_29 = Embedding(input_dim=emb_mtx_f1_29.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_29],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_29_f1)
    x1_29 = Flatten()(x1_29)

    x2_29 = Embedding(input_dim=emb_mtx_f2_29.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_29],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_29_f2)
    x2_29 = Flatten()(x2_29)

    concat_29 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_29 = Dense(10, activation='elu',)(concat_29)
#!################################################################################################################
    input_30_f1 = Input(shape=(1,), name='input_30_f1_layer')
    input_30_f2 = Input(shape=(1,), name='input_30_f2_layer')

    x1_30 = Embedding(input_dim=emb_mtx_f1_30.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_30],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_30_f1)
    x1_30 = Flatten()(x1_30)

    x2_30 = Embedding(input_dim=emb_mtx_f2_30.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_30],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_30_f2)
    x2_30 = Flatten()(x2_30)

    concat_30 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_30 = Dense(10, activation='elu',)(concat_30)
#!################################################################################################################
    input_31_f1 = Input(shape=(1,), name='input_31_f1_layer')
    input_31_f2 = Input(shape=(1,), name='input_31_f2_layer')

    x1_31 = Embedding(input_dim=emb_mtx_f1_31.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_31],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_31_f1)
    x1_31 = Flatten()(x1_31)

    x2_31 = Embedding(input_dim=emb_mtx_f2_31.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_31],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_31_f2)
    x2_31 = Flatten()(x2_31)

    concat_31 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_31 = Dense(10, activation='elu',)(concat_31)

#!################################################################################################################
    input_32_f1 = Input(shape=(1,), name='input_32_f1_layer')
    input_32_f2 = Input(shape=(1,), name='input_32_f2_layer')

    x1_32 = Embedding(input_dim=emb_mtx_f1_32.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_32],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_32_f1)
    x1_32 = Flatten()(x1_32)

    x2_32 = Embedding(input_dim=emb_mtx_f2_32.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_32],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_32_f2)
    x2_32 = Flatten()(x2_32)

    concat_32 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_32 = Dense(10, activation='elu',)(concat_32)
#!################################################################################################################
    input_33_f1 = Input(shape=(1,), name='input_33_f1_layer')
    input_33_f2 = Input(shape=(1,), name='input_33_f2_layer')

    x1_33 = Embedding(input_dim=emb_mtx_f1_33.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_33],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_33_f1)
    x1_33 = Flatten()(x1_33)

    x2_33 = Embedding(input_dim=emb_mtx_f2_33.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_33],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_33_f2)
    x2_33 = Flatten()(x2_33)

    concat_33 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_33 = Dense(10, activation='elu',)(concat_33)
#!################################################################################################################
    input_34_f1 = Input(shape=(1,), name='input_34_f1_layer')
    input_34_f2 = Input(shape=(1,), name='input_34_f2_layer')

    x1_34 = Embedding(input_dim=emb_mtx_f1_34.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_34],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_34_f1)
    x1_34 = Flatten()(x1_34)

    x2_34 = Embedding(input_dim=emb_mtx_f2_34.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_34],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_34_f2)
    x2_34 = Flatten()(x2_34)

    concat_34 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_34 = Dense(10, activation='elu',)(concat_34)
#!################################################################################################################
    input_35_f1 = Input(shape=(1,), name='input_35_f1_layer')
    input_35_f2 = Input(shape=(1,), name='input_35_f2_layer')

    x1_35 = Embedding(input_dim=emb_mtx_f1_35.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f1_35],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_35_f1)
    x1_35 = Flatten()(x1_35)

    x2_35 = Embedding(input_dim=emb_mtx_f2_35.shape[0],
                      output_dim=64,
                      weights=[emb_mtx_f2_35],
                      trainable=False,
                      input_length=1,
                      mask_zero=True)(input_35_f2)
    x2_35 = Flatten()(x2_35)

    concat_35 = concatenate([x1_1, x2_1], axis=-1)

    output_f1_f2_35 = Dense(10, activation='elu',)(concat_35)
#!################################################################################################################

    f1_f2_output = concatenate([output_f1_f2_1, output_f1_f2_2], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_3], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_4], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_5], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_6], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_7], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_8], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_9], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_10], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_11], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_12], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_13], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_14], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_15], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_16], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_17], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_18], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_19], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_20], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_21], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_22], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_23], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_24], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_25], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_26], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_27], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_28], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_29], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_30], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_31], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_32], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_33], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_34], axis=-1)
    f1_f2_output = concatenate([f1_f2_output, output_f1_f2_35], axis=-1)

#!################################################################################################################
    inputs_all = [input_1_f1, input_1_f2,
                  input_2_f1, input_2_f2,
                  input_3_f1, input_3_f2,
                  input_4_f1, input_4_f2,
                  input_5_f1, input_5_f2,
                  input_6_f1, input_6_f2,
                  input_7_f1, input_7_f2,
                  input_8_f1, input_8_f2,
                  input_9_f1, input_9_f2,
                  input_10_f1, input_10_f2,
                  input_11_f1, input_11_f2,
                  input_12_f1, input_12_f2,
                  input_13_f1, input_13_f2,
                  input_14_f1, input_14_f2,
                  input_15_f1, input_15_f2,
                  input_16_f1, input_16_f2,
                  input_17_f1, input_17_f2,
                  input_18_f1, input_18_f2,
                  input_19_f1, input_19_f2,
                  input_20_f1, input_20_f2,
                  input_21_f1, input_21_f2,
                  input_22_f1, input_22_f2,
                  input_23_f1, input_23_f2,
                  input_24_f1, input_24_f2,
                  input_25_f1, input_25_f2,
                  input_26_f1, input_26_f2,
                  input_27_f1, input_27_f2,
                  input_28_f1, input_28_f2,
                  input_29_f1, input_29_f2,
                  input_30_f1, input_30_f2,
                  input_31_f1, input_31_f2,
                  input_32_f1, input_32_f2,
                  input_33_f1, input_33_f2,
                  input_34_f1, input_34_f2,
                  input_35_f1, input_35_f2]
    seq_inputs_dict = get_seq_input_layers(cols=arr_name_list)
    inputs_all = list(seq_inputs_dict.values())+inputs_all  # 输入层list
    # slc
    # masks = tf.equal(seq_inputs_dict['task_id'], 0)
    import pdb
    pdb.set_trace()
    masks = tf.equal(seq_inputs_dict['label'], 0)
    # 普通序列+label序列
    layers2concat = []
    for index, col in enumerate(arr_name_list):
        print(col, 'get embedding!')
        emb_layer = get_emb_layer(
            col, trainable=TRAINABLE_DICT[col], emb_matrix=id_list_dict_emb_all[col][1])
        x = emb_layer(seq_inputs_dict[col])
        if conv1d_info_dict[col] > -1:
            cov_layer = tf.keras.layers.Conv1D(filters=conv1d_info_dict[col],
                                               kernel_size=1,
                                               activation='relu')
            x = cov_layer(x)
        layers2concat.append(x)
    x = tf.keras.layers.concatenate(layers2concat)
#!################################################################################################################

    x = trans_net(x, masks, hidden_unit=256)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()
    average_pool = tf.keras.layers.GlobalAveragePooling1D()
    xmaxpool = max_pool(x)
    xmeanpool = average_pool(x)

    trans_output = tf.keras.layers.concatenate([xmaxpool, xmeanpool])

#!################################################################################################################
#!mix1
    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)
    inputs_all = inputs_all+[features]
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    mix = concatenate([trans_output, f1_f2_output,
                       dnn_input], axis=-1)  # !#mix

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(mix)

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

#!################################################################################################################

    model = Model(inputs=inputs_all,
                  outputs=[output])

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=multi_category_focal_loss2(
            alpha=0.35),
        metrics=['AUC'])
    print(model.summary())
    return model


# ## user_item交叉特征的输入定义 总计35对item,user属性做交叉

# In[5]:


f1_f2_list = [['task_id', 'age'], ['task_id', 'city'], ['task_id', 'city_rank'],
              ['task_id', 'device_name'], [
                  'task_id', 'career'], ['task_id', 'gender'],
              ['task_id', 'residence'], ['adv_id', 'age'], [
                  'adv_id', 'city'], ['adv_id', 'city_rank'],
              ['adv_id', 'device_name'], [
                  'adv_id', 'career'], ['adv_id', 'gender'],
              ['adv_id', 'residence'], ['creat_type_cd', 'age'], [
                  'creat_type_cd', 'city'],
              ['creat_type_cd', 'city_rank'], ['creat_type_cd',
                                               'device_name'], ['creat_type_cd', 'career'],
              ['creat_type_cd', 'gender'], [
                  'creat_type_cd', 'residence'], ['indu_name', 'age'],
              ['indu_name', 'city'], ['indu_name', 'city_rank'], [
                  'indu_name', 'device_name'],
              ['indu_name', 'career'], ['indu_name', 'gender'], [
                  'indu_name', 'residence'],
              ['adv_prim_id', 'age'], ['adv_prim_id', 'city'], [
                  'adv_prim_id', 'city_rank'],
              ['adv_prim_id', 'device_name'], ['adv_prim_id', 'career'], ['adv_prim_id', 'gender'], ['adv_prim_id', 'residence']]

emb_mtx_f1_path = []
for i in f1_f2_list:
    emb_mtx_f1_path.append(
        str(i[0])+'_'+str(i[1])+'_emb_mtx_adv_userseq_adv.npy')
emb_mtx_f1_1 = np.load(emb_mtx_f1_path[0], allow_pickle=True)
emb_mtx_f1_2 = np.load(emb_mtx_f1_path[1], allow_pickle=True)
emb_mtx_f1_3 = np.load(emb_mtx_f1_path[2], allow_pickle=True)
emb_mtx_f1_4 = np.load(emb_mtx_f1_path[3], allow_pickle=True)
emb_mtx_f1_5 = np.load(emb_mtx_f1_path[4], allow_pickle=True)
emb_mtx_f1_6 = np.load(emb_mtx_f1_path[5], allow_pickle=True)
emb_mtx_f1_7 = np.load(emb_mtx_f1_path[6], allow_pickle=True)
emb_mtx_f1_8 = np.load(emb_mtx_f1_path[7], allow_pickle=True)
emb_mtx_f1_9 = np.load(emb_mtx_f1_path[8], allow_pickle=True)
emb_mtx_f1_10 = np.load(emb_mtx_f1_path[9], allow_pickle=True)
emb_mtx_f1_11 = np.load(emb_mtx_f1_path[10], allow_pickle=True)
emb_mtx_f1_12 = np.load(emb_mtx_f1_path[11], allow_pickle=True)
emb_mtx_f1_13 = np.load(emb_mtx_f1_path[12], allow_pickle=True)
emb_mtx_f1_14 = np.load(emb_mtx_f1_path[13], allow_pickle=True)
emb_mtx_f1_15 = np.load(emb_mtx_f1_path[14], allow_pickle=True)
emb_mtx_f1_16 = np.load(emb_mtx_f1_path[15], allow_pickle=True)
emb_mtx_f1_17 = np.load(emb_mtx_f1_path[16], allow_pickle=True)
emb_mtx_f1_18 = np.load(emb_mtx_f1_path[17], allow_pickle=True)
emb_mtx_f1_19 = np.load(emb_mtx_f1_path[18], allow_pickle=True)
emb_mtx_f1_20 = np.load(emb_mtx_f1_path[19], allow_pickle=True)
emb_mtx_f1_21 = np.load(emb_mtx_f1_path[20], allow_pickle=True)
emb_mtx_f1_22 = np.load(emb_mtx_f1_path[21], allow_pickle=True)
emb_mtx_f1_23 = np.load(emb_mtx_f1_path[22], allow_pickle=True)
emb_mtx_f1_24 = np.load(emb_mtx_f1_path[23], allow_pickle=True)
emb_mtx_f1_25 = np.load(emb_mtx_f1_path[24], allow_pickle=True)
emb_mtx_f1_26 = np.load(emb_mtx_f1_path[25], allow_pickle=True)
emb_mtx_f1_27 = np.load(emb_mtx_f1_path[26], allow_pickle=True)
emb_mtx_f1_28 = np.load(emb_mtx_f1_path[27], allow_pickle=True)
emb_mtx_f1_29 = np.load(emb_mtx_f1_path[28], allow_pickle=True)
emb_mtx_f1_30 = np.load(emb_mtx_f1_path[29], allow_pickle=True)
emb_mtx_f1_31 = np.load(emb_mtx_f1_path[30], allow_pickle=True)
emb_mtx_f1_32 = np.load(emb_mtx_f1_path[31], allow_pickle=True)
emb_mtx_f1_33 = np.load(emb_mtx_f1_path[32], allow_pickle=True)
emb_mtx_f1_34 = np.load(emb_mtx_f1_path[33], allow_pickle=True)
emb_mtx_f1_35 = np.load(emb_mtx_f1_path[34], allow_pickle=True)

emb_mtx_f2_path = []
for i in f1_f2_list:
    emb_mtx_f2_path.append(
        str(i[0])+'_'+str(i[1])+'_emb_mtx_adv_userseq_user.npy')
emb_mtx_f2_1 = np.load(emb_mtx_f2_path[0], allow_pickle=True)
emb_mtx_f2_2 = np.load(emb_mtx_f2_path[1], allow_pickle=True)
emb_mtx_f2_3 = np.load(emb_mtx_f2_path[2], allow_pickle=True)
emb_mtx_f2_4 = np.load(emb_mtx_f2_path[3], allow_pickle=True)
emb_mtx_f2_5 = np.load(emb_mtx_f2_path[4], allow_pickle=True)
emb_mtx_f2_6 = np.load(emb_mtx_f2_path[5], allow_pickle=True)
emb_mtx_f2_7 = np.load(emb_mtx_f2_path[6], allow_pickle=True)
emb_mtx_f2_8 = np.load(emb_mtx_f2_path[7], allow_pickle=True)
emb_mtx_f2_9 = np.load(emb_mtx_f2_path[8], allow_pickle=True)
emb_mtx_f2_10 = np.load(emb_mtx_f2_path[9], allow_pickle=True)
emb_mtx_f2_11 = np.load(emb_mtx_f2_path[10], allow_pickle=True)
emb_mtx_f2_12 = np.load(emb_mtx_f2_path[11], allow_pickle=True)
emb_mtx_f2_13 = np.load(emb_mtx_f2_path[12], allow_pickle=True)
emb_mtx_f2_14 = np.load(emb_mtx_f2_path[13], allow_pickle=True)
emb_mtx_f2_15 = np.load(emb_mtx_f2_path[14], allow_pickle=True)
emb_mtx_f2_16 = np.load(emb_mtx_f2_path[15], allow_pickle=True)
emb_mtx_f2_17 = np.load(emb_mtx_f2_path[16], allow_pickle=True)
emb_mtx_f2_18 = np.load(emb_mtx_f2_path[17], allow_pickle=True)
emb_mtx_f2_19 = np.load(emb_mtx_f2_path[18], allow_pickle=True)
emb_mtx_f2_20 = np.load(emb_mtx_f2_path[19], allow_pickle=True)
emb_mtx_f2_21 = np.load(emb_mtx_f2_path[20], allow_pickle=True)
emb_mtx_f2_22 = np.load(emb_mtx_f2_path[21], allow_pickle=True)
emb_mtx_f2_23 = np.load(emb_mtx_f2_path[22], allow_pickle=True)
emb_mtx_f2_24 = np.load(emb_mtx_f2_path[23], allow_pickle=True)
emb_mtx_f2_25 = np.load(emb_mtx_f2_path[24], allow_pickle=True)
emb_mtx_f2_26 = np.load(emb_mtx_f2_path[25], allow_pickle=True)
emb_mtx_f2_27 = np.load(emb_mtx_f2_path[26], allow_pickle=True)
emb_mtx_f2_28 = np.load(emb_mtx_f2_path[27], allow_pickle=True)
emb_mtx_f2_29 = np.load(emb_mtx_f2_path[28], allow_pickle=True)
emb_mtx_f2_30 = np.load(emb_mtx_f2_path[29], allow_pickle=True)
emb_mtx_f2_31 = np.load(emb_mtx_f2_path[30], allow_pickle=True)
emb_mtx_f2_32 = np.load(emb_mtx_f2_path[31], allow_pickle=True)
emb_mtx_f2_33 = np.load(emb_mtx_f2_path[32], allow_pickle=True)
emb_mtx_f2_34 = np.load(emb_mtx_f2_path[33], allow_pickle=True)
emb_mtx_f2_35 = np.load(emb_mtx_f2_path[34], allow_pickle=True)

w2v_f1_train_path_list = []
for i in f1_f2_list:
    w2v_f1_train_path_list.append(
        str(i[0])+'_'+str(i[1])+'_adv_userseq_adv_train.npy')
w2v_1_f1_train = np.load(w2v_f1_train_path_list[0])
w2v_2_f1_train = np.load(w2v_f1_train_path_list[1])
w2v_3_f1_train = np.load(w2v_f1_train_path_list[2])
w2v_4_f1_train = np.load(w2v_f1_train_path_list[3])
w2v_5_f1_train = np.load(w2v_f1_train_path_list[4])
w2v_6_f1_train = np.load(w2v_f1_train_path_list[5])
w2v_7_f1_train = np.load(w2v_f1_train_path_list[6])
w2v_8_f1_train = np.load(w2v_f1_train_path_list[7])
w2v_9_f1_train = np.load(w2v_f1_train_path_list[8])
w2v_10_f1_train = np.load(w2v_f1_train_path_list[9])
w2v_11_f1_train = np.load(w2v_f1_train_path_list[10])
w2v_12_f1_train = np.load(w2v_f1_train_path_list[11])
w2v_13_f1_train = np.load(w2v_f1_train_path_list[12])
w2v_14_f1_train = np.load(w2v_f1_train_path_list[13])
w2v_15_f1_train = np.load(w2v_f1_train_path_list[14])
w2v_16_f1_train = np.load(w2v_f1_train_path_list[15])
w2v_17_f1_train = np.load(w2v_f1_train_path_list[16])
w2v_18_f1_train = np.load(w2v_f1_train_path_list[17])
w2v_19_f1_train = np.load(w2v_f1_train_path_list[18])
w2v_20_f1_train = np.load(w2v_f1_train_path_list[19])
w2v_21_f1_train = np.load(w2v_f1_train_path_list[20])
w2v_22_f1_train = np.load(w2v_f1_train_path_list[21])
w2v_23_f1_train = np.load(w2v_f1_train_path_list[22])
w2v_24_f1_train = np.load(w2v_f1_train_path_list[23])
w2v_25_f1_train = np.load(w2v_f1_train_path_list[24])
w2v_26_f1_train = np.load(w2v_f1_train_path_list[25])
w2v_27_f1_train = np.load(w2v_f1_train_path_list[26])
w2v_28_f1_train = np.load(w2v_f1_train_path_list[27])
w2v_29_f1_train = np.load(w2v_f1_train_path_list[28])
w2v_30_f1_train = np.load(w2v_f1_train_path_list[29])
w2v_31_f1_train = np.load(w2v_f1_train_path_list[30])
w2v_32_f1_train = np.load(w2v_f1_train_path_list[31])
w2v_33_f1_train = np.load(w2v_f1_train_path_list[32])
w2v_34_f1_train = np.load(w2v_f1_train_path_list[33])
w2v_35_f1_train = np.load(w2v_f1_train_path_list[34])

w2v_f2_train_path_list = []
for i in f1_f2_list:
    w2v_f2_train_path_list.append(
        str(i[0])+'_'+str(i[1])+'_adv_userseq_user_train.npy')
w2v_1_f2_train = np.load(w2v_f2_train_path_list[0])
w2v_2_f2_train = np.load(w2v_f2_train_path_list[1])
w2v_3_f2_train = np.load(w2v_f2_train_path_list[2])
w2v_4_f2_train = np.load(w2v_f2_train_path_list[3])
w2v_5_f2_train = np.load(w2v_f2_train_path_list[4])
w2v_6_f2_train = np.load(w2v_f2_train_path_list[5])
w2v_7_f2_train = np.load(w2v_f2_train_path_list[6])
w2v_8_f2_train = np.load(w2v_f2_train_path_list[7])
w2v_9_f2_train = np.load(w2v_f2_train_path_list[8])
w2v_10_f2_train = np.load(w2v_f2_train_path_list[9])
w2v_11_f2_train = np.load(w2v_f2_train_path_list[10])
w2v_12_f2_train = np.load(w2v_f2_train_path_list[11])
w2v_13_f2_train = np.load(w2v_f2_train_path_list[12])
w2v_14_f2_train = np.load(w2v_f2_train_path_list[13])
w2v_15_f2_train = np.load(w2v_f2_train_path_list[14])
w2v_16_f2_train = np.load(w2v_f2_train_path_list[15])
w2v_17_f2_train = np.load(w2v_f2_train_path_list[16])
w2v_18_f2_train = np.load(w2v_f2_train_path_list[17])
w2v_19_f2_train = np.load(w2v_f2_train_path_list[18])
w2v_20_f2_train = np.load(w2v_f2_train_path_list[19])
w2v_21_f2_train = np.load(w2v_f2_train_path_list[20])
w2v_22_f2_train = np.load(w2v_f2_train_path_list[21])
w2v_23_f2_train = np.load(w2v_f2_train_path_list[22])
w2v_24_f2_train = np.load(w2v_f2_train_path_list[23])
w2v_25_f2_train = np.load(w2v_f2_train_path_list[24])
w2v_26_f2_train = np.load(w2v_f2_train_path_list[25])
w2v_27_f2_train = np.load(w2v_f2_train_path_list[26])
w2v_28_f2_train = np.load(w2v_f2_train_path_list[27])
w2v_29_f2_train = np.load(w2v_f2_train_path_list[28])
w2v_30_f2_train = np.load(w2v_f2_train_path_list[29])
w2v_31_f2_train = np.load(w2v_f2_train_path_list[30])
w2v_32_f2_train = np.load(w2v_f2_train_path_list[31])
w2v_33_f2_train = np.load(w2v_f2_train_path_list[32])
w2v_34_f2_train = np.load(w2v_f2_train_path_list[33])
w2v_35_f2_train = np.load(w2v_f2_train_path_list[34])

w2v_f1_test_path_list = []
for i in f1_f2_list:
    w2v_f1_test_path_list.append(
        str(i[0])+'_'+str(i[1])+'_adv_userseq_adv_test.npy')
w2v_1_f1_test = np.load(w2v_f1_test_path_list[0])
w2v_2_f1_test = np.load(w2v_f1_test_path_list[1])
w2v_3_f1_test = np.load(w2v_f1_test_path_list[2])
w2v_4_f1_test = np.load(w2v_f1_test_path_list[3])
w2v_5_f1_test = np.load(w2v_f1_test_path_list[4])
w2v_6_f1_test = np.load(w2v_f1_test_path_list[5])
w2v_7_f1_test = np.load(w2v_f1_test_path_list[6])
w2v_8_f1_test = np.load(w2v_f1_test_path_list[7])
w2v_9_f1_test = np.load(w2v_f1_test_path_list[8])
w2v_10_f1_test = np.load(w2v_f1_test_path_list[9])
w2v_11_f1_test = np.load(w2v_f1_test_path_list[10])
w2v_12_f1_test = np.load(w2v_f1_test_path_list[11])
w2v_13_f1_test = np.load(w2v_f1_test_path_list[12])
w2v_14_f1_test = np.load(w2v_f1_test_path_list[13])
w2v_15_f1_test = np.load(w2v_f1_test_path_list[14])
w2v_16_f1_test = np.load(w2v_f1_test_path_list[15])
w2v_17_f1_test = np.load(w2v_f1_test_path_list[16])
w2v_18_f1_test = np.load(w2v_f1_test_path_list[17])
w2v_19_f1_test = np.load(w2v_f1_test_path_list[18])
w2v_20_f1_test = np.load(w2v_f1_test_path_list[19])
w2v_21_f1_test = np.load(w2v_f1_test_path_list[20])
w2v_22_f1_test = np.load(w2v_f1_test_path_list[21])
w2v_23_f1_test = np.load(w2v_f1_test_path_list[22])
w2v_24_f1_test = np.load(w2v_f1_test_path_list[23])
w2v_25_f1_test = np.load(w2v_f1_test_path_list[24])
w2v_26_f1_test = np.load(w2v_f1_test_path_list[25])
w2v_27_f1_test = np.load(w2v_f1_test_path_list[26])
w2v_28_f1_test = np.load(w2v_f1_test_path_list[27])
w2v_29_f1_test = np.load(w2v_f1_test_path_list[28])
w2v_30_f1_test = np.load(w2v_f1_test_path_list[29])
w2v_31_f1_test = np.load(w2v_f1_test_path_list[30])
w2v_32_f1_test = np.load(w2v_f1_test_path_list[31])
w2v_33_f1_test = np.load(w2v_f1_test_path_list[32])
w2v_34_f1_test = np.load(w2v_f1_test_path_list[33])
w2v_35_f1_test = np.load(w2v_f1_test_path_list[34])

w2v_f2_test_path_list = []
for i in f1_f2_list:
    w2v_f2_test_path_list.append(
        str(i[0])+'_'+str(i[1])+'_adv_userseq_user_test.npy')
w2v_1_f2_test = np.load(w2v_f2_test_path_list[0])
w2v_2_f2_test = np.load(w2v_f2_test_path_list[1])
w2v_3_f2_test = np.load(w2v_f2_test_path_list[2])
w2v_4_f2_test = np.load(w2v_f2_test_path_list[3])
w2v_5_f2_test = np.load(w2v_f2_test_path_list[4])
w2v_6_f2_test = np.load(w2v_f2_test_path_list[5])
w2v_7_f2_test = np.load(w2v_f2_test_path_list[6])
w2v_8_f2_test = np.load(w2v_f2_test_path_list[7])
w2v_9_f2_test = np.load(w2v_f2_test_path_list[8])
w2v_10_f2_test = np.load(w2v_f2_test_path_list[9])
w2v_11_f2_test = np.load(w2v_f2_test_path_list[10])
w2v_12_f2_test = np.load(w2v_f2_test_path_list[11])
w2v_13_f2_test = np.load(w2v_f2_test_path_list[12])
w2v_14_f2_test = np.load(w2v_f2_test_path_list[13])
w2v_15_f2_test = np.load(w2v_f2_test_path_list[14])
w2v_16_f2_test = np.load(w2v_f2_test_path_list[15])
w2v_17_f2_test = np.load(w2v_f2_test_path_list[16])
w2v_18_f2_test = np.load(w2v_f2_test_path_list[17])
w2v_19_f2_test = np.load(w2v_f2_test_path_list[18])
w2v_20_f2_test = np.load(w2v_f2_test_path_list[19])
w2v_21_f2_test = np.load(w2v_f2_test_path_list[20])
w2v_22_f2_test = np.load(w2v_f2_test_path_list[21])
w2v_23_f2_test = np.load(w2v_f2_test_path_list[22])
w2v_24_f2_test = np.load(w2v_f2_test_path_list[23])
w2v_25_f2_test = np.load(w2v_f2_test_path_list[24])
w2v_26_f2_test = np.load(w2v_f2_test_path_list[25])
w2v_27_f2_test = np.load(w2v_f2_test_path_list[26])
w2v_28_f2_test = np.load(w2v_f2_test_path_list[27])
w2v_29_f2_test = np.load(w2v_f2_test_path_list[28])
w2v_30_f2_test = np.load(w2v_f2_test_path_list[29])
w2v_31_f2_test = np.load(w2v_f2_test_path_list[30])
w2v_32_f2_test = np.load(w2v_f2_test_path_list[31])
w2v_33_f2_test = np.load(w2v_f2_test_path_list[32])
w2v_34_f2_test = np.load(w2v_f2_test_path_list[33])
w2v_35_f2_test = np.load(w2v_f2_test_path_list[34])


# ## 基础特征输入定义

# In[6]:


sparse_features = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 'age', 'city_rank',
                   'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career',
                   'gender', 'net_type', 'residence', 'emui_dev', 'indu_name', ]
#    'communication_onlinerate_1', 'communication_onlinerate_2', 'communication_onlinerate_3',
#    'communication_onlinerate_4', 'communication_onlinerate_5', 'communication_onlinerate_6',
#    'communication_onlinerate_7', 'communication_onlinerate_8', 'communication_onlinerate_9',
#    'communication_onlinerate_10', 'communication_onlinerate_11', 'communication_onlinerate_12',
#    'communication_onlinerate_13', 'communication_onlinerate_14', 'communication_onlinerate_15',
#    'communication_onlinerate_16', 'communication_onlinerate_17', 'communication_onlinerate_18',
#    'communication_onlinerate_19', 'communication_onlinerate_20', 'communication_onlinerate_21',
#    'communication_onlinerate_22', 'communication_onlinerate_23']  # e.g.:05db9164
dense_features = ['device_size', 'his_app_size', 'his_on_shelf_time', 'list_time', 'device_price', 'up_life_duration',
                  'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_avgonline_30d', 'task_id_count',
                  'task_id_pt_d_count', 'adv_id_count', 'adv_id_pt_d_count',
                  'creat_type_cd_count', 'creat_type_cd_pt_d_count',
                  'adv_prim_id_count', 'adv_prim_id_pt_d_count', 'dev_id_count',
                  'dev_id_pt_d_count', 'inter_type_cd_count',
                  'inter_type_cd_pt_d_count', 'slot_id_count', 'slot_id_pt_d_count',
                  'spread_app_id_count', 'spread_app_id_pt_d_count', 'tags_count',
                  'tags_pt_d_count', 'app_first_class_count',
                  'app_first_class_pt_d_count', 'app_second_class_count',
                  'app_second_class_pt_d_count', 'city_count', 'city_pt_d_count',
                  'device_name_count', 'device_name_pt_d_count', 'career_count',
                  'career_pt_d_count', 'gender_count', 'gender_pt_d_count',
                  'age_count', 'age_pt_d_count', 'net_type_count',
                  'net_type_pt_d_count', 'residence_count', 'residence_pt_d_count',
                  'emui_dev_count', 'emui_dev_pt_d_count', 'indu_name_count',
                  'indu_name_pt_d_count',
                  #   'communication_onlinerate_1_count',
                  #   'communication_onlinerate_1_pt_d_count',
                  #   'communication_onlinerate_2_count',
                  #   'communication_onlinerate_2_pt_d_count',
                  #   'communication_onlinerate_3_count',
                  #   'communication_onlinerate_3_pt_d_count',
                  #   'communication_onlinerate_4_count',
                  #   'communication_onlinerate_4_pt_d_count',
                  #   'communication_onlinerate_5_count',
                  #   'communication_onlinerate_5_pt_d_count',
                  #   'communication_onlinerate_6_count',
                  #   'communication_onlinerate_6_pt_d_count',
                  #   'communication_onlinerate_7_count',
                  #   'communication_onlinerate_7_pt_d_count',
                  #   'communication_onlinerate_8_count',
                  #   'communication_onlinerate_8_pt_d_count',
                  #   'communication_onlinerate_9_count',
                  #   'communication_onlinerate_9_pt_d_count',
                  #   'communication_onlinerate_10_count',
                  #   'communication_onlinerate_10_pt_d_count',
                  #   'communication_onlinerate_11_count',
                  #   'communication_onlinerate_11_pt_d_count',
                  #   'communication_onlinerate_12_count',
                  #   'communication_onlinerate_12_pt_d_count',
                  #   'communication_onlinerate_13_count',
                  #   'communication_onlinerate_13_pt_d_count',
                  #   'communication_onlinerate_14_count',
                  #   'communication_onlinerate_14_pt_d_count',
                  #   'communication_onlinerate_15_count',
                  #   'communication_onlinerate_15_pt_d_count',
                  #   'communication_onlinerate_16_count',
                  #   'communication_onlinerate_16_pt_d_count',
                  #   'communication_onlinerate_17_count',
                  #   'communication_onlinerate_17_pt_d_count',
                  #   'communication_onlinerate_18_count',
                  #   'communication_onlinerate_18_pt_d_count',
                  #   'communication_onlinerate_19_count',
                  #   'communication_onlinerate_19_pt_d_count',
                  #   'communication_onlinerate_20_count',
                  #   'communication_onlinerate_20_pt_d_count',
                  #   'communication_onlinerate_21_count',
                  #   'communication_onlinerate_21_pt_d_count',
                  #   'communication_onlinerate_22_count',
                  #   'communication_onlinerate_22_pt_d_count',
                  #   'communication_onlinerate_23_count',
                  #   'communication_onlinerate_23_pt_d_count',
                  'uidtask_id_nunique',
                  'uidtask_id_pt_d_nunique', 'uidadv_id_nunique',
                  'uidadv_id_pt_d_nunique', 'uiddev_id_nunique',
                  'uiddev_id_pt_d_nunique', 'uidspread_app_id_nunique',
                  'uidspread_app_id_pt_d_nunique', 'uidindu_name_nunique',
                  'uidindu_name_pt_d_nunique', 'agetask_id_nunique',
                  'agetask_id_pt_d_nunique', 'ageadv_id_nunique',
                  'ageadv_id_pt_d_nunique', 'agedev_id_nunique',
                  'agedev_id_pt_d_nunique', 'agespread_app_id_nunique',
                  'agespread_app_id_pt_d_nunique', 'ageindu_name_nunique',
                  'ageindu_name_pt_d_nunique', 'gendertask_id_nunique',
                  'gendertask_id_pt_d_nunique', 'genderadv_id_nunique',
                  'genderadv_id_pt_d_nunique', 'genderdev_id_nunique',
                  'genderdev_id_pt_d_nunique', 'genderspread_app_id_nunique',
                  'genderspread_app_id_pt_d_nunique', 'genderindu_name_nunique',
                  'genderindu_name_pt_d_nunique', 'careertask_id_nunique',
                  'careertask_id_pt_d_nunique', 'careeradv_id_nunique',
                  'careeradv_id_pt_d_nunique', 'careerdev_id_nunique',
                  'careerdev_id_pt_d_nunique', 'careerspread_app_id_nunique',
                  'careerspread_app_id_pt_d_nunique',
                  'careerindu_name_pt_d_nunique', 'citytask_id_nunique',
                  'citytask_id_pt_d_nunique', 'cityadv_id_nunique',
                  'cityadv_id_pt_d_nunique', 'citydev_id_nunique',
                  'citydev_id_pt_d_nunique', 'cityspread_app_id_nunique',
                  'cityspread_app_id_pt_d_nunique', 'cityindu_name_nunique',
                  'cityindu_name_pt_d_nunique', 'slot_idtask_id_nunique',
                  'slot_idtask_id_pt_d_nunique', 'slot_idadv_id_nunique',
                  'slot_idadv_id_pt_d_nunique', 'slot_iddev_id_nunique',
                  'slot_iddev_id_pt_d_nunique', 'slot_idspread_app_id_nunique',
                  'slot_idspread_app_id_pt_d_nunique', 'slot_idindu_name_nunique',
                  'slot_idindu_name_pt_d_nunique', 'net_typetask_id_nunique',
                  'net_typetask_id_pt_d_nunique', 'net_typeadv_id_nunique',
                  'net_typeadv_id_pt_d_nunique', 'net_typedev_id_nunique',
                  'net_typedev_id_pt_d_nunique', 'net_typespread_app_id_nunique',
                  'net_typespread_app_id_pt_d_nunique', 'net_typeindu_name_nunique',
                  'net_typeindu_name_pt_d_nunique', 'uidtask_id_nunique_target_enc',
                  'uidtask_id_pt_d_nunique_target_enc',
                  'uidadv_id_nunique_target_enc',
                  'uidadv_id_pt_d_nunique_target_enc',
                  'uiddev_id_nunique_target_enc',
                  'uiddev_id_pt_d_nunique_target_enc',
                  'uidspread_app_id_nunique_target_enc',
                  'uidspread_app_id_pt_d_nunique_target_enc',
                  'uidindu_name_nunique_target_enc',
                  'uidindu_name_pt_d_nunique_target_enc',
                  'agetask_id_nunique_target_enc',
                  'agetask_id_pt_d_nunique_target_enc',
                  'ageadv_id_nunique_target_enc',
                  'ageadv_id_pt_d_nunique_target_enc',
                  'agedev_id_nunique_target_enc',
                  'agedev_id_pt_d_nunique_target_enc',
                  'agespread_app_id_nunique_target_enc',
                  'agespread_app_id_pt_d_nunique_target_enc',
                  'ageindu_name_nunique_target_enc',
                  'ageindu_name_pt_d_nunique_target_enc',
                  'gendertask_id_nunique_target_enc',
                  'gendertask_id_pt_d_nunique_target_enc',
                  'genderadv_id_nunique_target_enc',
                  'genderadv_id_pt_d_nunique_target_enc',
                  'genderdev_id_nunique_target_enc',
                  'genderdev_id_pt_d_nunique_target_enc',
                  'genderspread_app_id_nunique_target_enc',
                  'genderspread_app_id_pt_d_nunique_target_enc',
                  'genderindu_name_nunique_target_enc',
                  'genderindu_name_pt_d_nunique_target_enc',
                  'careertask_id_nunique_target_enc',
                  'careertask_id_pt_d_nunique_target_enc',
                  'careeradv_id_nunique_target_enc',
                  'careeradv_id_pt_d_nunique_target_enc',
                  'careerdev_id_nunique_target_enc',
                  'careerdev_id_pt_d_nunique_target_enc',
                  'careerspread_app_id_nunique_target_enc',
                  'careerspread_app_id_pt_d_nunique_target_enc',
                  'careerindu_name_pt_d_nunique_target_enc',
                  'citytask_id_nunique_target_enc',
                  'citytask_id_pt_d_nunique_target_enc',
                  'cityadv_id_nunique_target_enc',
                  'cityadv_id_pt_d_nunique_target_enc',
                  'citydev_id_nunique_target_enc',
                  'citydev_id_pt_d_nunique_target_enc',
                  'cityspread_app_id_nunique_target_enc',
                  'cityspread_app_id_pt_d_nunique_target_enc',
                  'cityindu_name_nunique_target_enc',
                  'cityindu_name_pt_d_nunique_target_enc',
                  'slot_idtask_id_nunique_target_enc',
                  'slot_idtask_id_pt_d_nunique_target_enc',
                  'slot_idadv_id_nunique_target_enc',
                  'slot_idadv_id_pt_d_nunique_target_enc',
                  'slot_iddev_id_nunique_target_enc',
                  'slot_iddev_id_pt_d_nunique_target_enc',
                  'slot_idspread_app_id_nunique_target_enc',
                  'slot_idspread_app_id_pt_d_nunique_target_enc',
                  'slot_idindu_name_nunique_target_enc',
                  'slot_idindu_name_pt_d_nunique_target_enc',
                  'net_typetask_id_nunique_target_enc',
                  'net_typetask_id_pt_d_nunique_target_enc',
                  'net_typeadv_id_nunique_target_enc',
                  'net_typeadv_id_pt_d_nunique_target_enc',
                  'net_typedev_id_nunique_target_enc',
                  'net_typedev_id_pt_d_nunique_target_enc',
                  'net_typespread_app_id_nunique_target_enc',
                  'net_typespread_app_id_pt_d_nunique_target_enc',
                  'net_typeindu_name_nunique_target_enc',
                  'net_typeindu_name_pt_d_nunique_target_enc', 'net_type_target_enc',
                  'task_id_target_enc', 'adv_id_target_enc',
                  'adv_prim_id_target_enc', 'age_target_enc',
                  'app_first_class_target_enc', 'app_second_class_target_enc',
                  'career_target_enc', 'city_target_enc',
                  'consume_purchase_target_enc', 'uid_target_enc',
                  'dev_id_target_enc', 'tags_target_enc', 'slot_id_target_enc']  # e.g.:16836.0

# deepfm_data[sparse_features] = deepfm_data[sparse_features].fillna('-1', )
# deepfm_data[dense_features] = deepfm_data[dense_features].fillna(0, )

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
# slc
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    deepfm_data[feat] = lbe.fit_transform(deepfm_data[feat])

for feat in tqdm(dense_features):
    mms = MinMaxScaler(feature_range=(0, 1))
    try:
        deepfm_data[feat] = mms.fit_transform(
            deepfm_data[feat].values.reshape(-1, 1)).astype(np.float32)
    except:
        print(f'{feat} missing !')
# del mms, lbe

# 2.count #unique features for each sparse field,and record dense feature field name

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=deepfm_data[feat].nunique(), embedding_dim=4)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                        for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input deepfm_data for model
# train, test = train_test_split(deepfm_data, test_size=0.2)
feature_names.append("raw_index")

deepfm_train_ = pd.DataFrame(
    {name: deepfm_data[name] for name in feature_names})
deepfm_train = deepfm_train_[deepfm_train_["raw_index"] < 6000_0000]


deepfm_test_ = pd.DataFrame(
    {name: deepfm_data[name] for name in feature_names})
deepfm_test = deepfm_test_[deepfm_test_["raw_index"] >= 6000_0000]

# 生成index
try:
    print(deepfm_data['id'].count())
except:
    deepfm_data['id'] = -111
deepfm_train_ = deepfm_data.loc[deepfm_data["raw_index"] < 6000_0000, [
    'id', 'label']]
del deepfm_test_
deepfm_data = deepfm_data[['raw_index']]
gc.collect()
feature_names.pop(-1)


# ## 对采样后的数据做10折 模型训练，结果取平均 10折分数在0.804左右，多个不同的采样，10折做融合 后续和其他队友模型融合

# In[7]:


# 10折

# callback
lr_list = [0.001, 0.001, 0.001, 0.0005, 0.00025,
           0.000125, 6.25e-05, 3.125e-05, 2e-05, 2e-05, 2e-05]


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
        checkpoint_dir, f"ckpt_zlhnn_model0929_addseq_{fold}_fold_{if_valid}")

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
        filename=f'./logs/model_zlhnn_model0929_addseq_{fold}_fold.log', separator=",", append=True)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_AUC',
                                                              factor=0.2,
                                                              patience=1,
                                                              min_delta=2e-4,
                                                              min_lr=1e-5)
    if if_valid:
        callbacks = [checkpoint_callback, csv_log_callback,
                     earlystop_callback, reduce_lr_callback]
    else:
        callbacks = [checkpoint_callback,
                     csv_log_callback,  reduce_lr_callback_trainall]
    return callbacks

# 没太搞清楚你test的id 目前这块用你的写法
# test_B = Cache.reload_cache('CACHE_test_B.pkl')
# test_B = test_B.reset_index()
# test_B.rename(columns={'index':'raw_index'},inplace=True)
# test_B['raw_index'] = test_B['raw_index']+6000_0000
# test_B_id = test_B[['raw_index','id']]
# deepfm_test_id = deepfm_data[deepfm_data["raw_index"]>=6000_0000]
# res_id = pd.merge(deepfm_test_id[['raw_index']],test_B_id,on='raw_index',how='left')['id']
# del test_B,test_B_id
# gc.collect()


# slc 暂时没有B榜
test_A = Cache.reload_cache('CACHE_test_A.pkl')
test_A = test_A.reset_index()
test_A.rename(columns={'index': 'raw_index'}, inplace=True)
test_A['raw_index'] = test_A['raw_index']+6000_0000
test_A_id = test_A[['raw_index', 'id']]
deepfm_test_id = deepfm_data[deepfm_data["raw_index"] >= 6000_0000]
res_id = pd.merge(deepfm_test_id[['raw_index']],
                  test_A_id, on='raw_index', how='left')['id']
del test_A, test_A_id
gc.collect()

res = pd.DataFrame(list(res_id.values), columns=['id'])
res['probability'] = 0
trainvalid = deepfm_train_[['id', 'label']]
trainvalid['probability'] = 0
random_state = 0
skf = StratifiedKFold(
    n_splits=10, random_state=random_state, shuffle=True)  # 抽90% 训练


# In[8]:


def get_input():
    input_train = {'input_1_f1_layer': w2v_1_f1_train,
                   'input_1_f2_layer': w2v_1_f2_train,
                   'input_2_f1_layer': w2v_2_f1_train,
                   'input_2_f2_layer': w2v_2_f2_train,
                   'input_3_f1_layer': w2v_3_f1_train,
                   'input_3_f2_layer': w2v_3_f2_train,
                   'input_4_f1_layer': w2v_4_f1_train,
                   'input_4_f2_layer': w2v_4_f2_train,
                   'input_5_f1_layer': w2v_5_f1_train,
                   'input_5_f2_layer': w2v_5_f2_train,
                   'input_6_f1_layer': w2v_6_f1_train,
                   'input_6_f2_layer': w2v_6_f2_train,
                   'input_7_f1_layer': w2v_7_f1_train,
                   'input_7_f2_layer': w2v_7_f2_train,
                   'input_8_f1_layer': w2v_8_f1_train,
                   'input_8_f2_layer': w2v_8_f2_train,
                   'input_9_f1_layer': w2v_9_f1_train,
                   'input_9_f2_layer': w2v_9_f2_train,
                   'input_10_f1_layer': w2v_10_f1_train,
                   'input_10_f2_layer': w2v_10_f2_train,
                   'input_11_f1_layer': w2v_11_f1_train,
                   'input_11_f2_layer': w2v_11_f2_train,
                   'input_12_f1_layer': w2v_12_f1_train,
                   'input_12_f2_layer': w2v_12_f2_train,
                   'input_13_f1_layer': w2v_13_f1_train,
                   'input_13_f2_layer': w2v_13_f2_train,
                   'input_14_f1_layer': w2v_14_f1_train,
                   'input_14_f2_layer': w2v_14_f2_train,
                   'input_15_f1_layer': w2v_15_f1_train,
                   'input_15_f2_layer': w2v_15_f2_train,
                   'input_16_f1_layer': w2v_16_f1_train,
                   'input_16_f2_layer': w2v_16_f2_train,
                   'input_17_f1_layer': w2v_17_f1_train,
                   'input_17_f2_layer': w2v_17_f2_train,
                   'input_18_f1_layer': w2v_18_f1_train,
                   'input_18_f2_layer': w2v_18_f2_train,
                   'input_19_f1_layer': w2v_19_f1_train,
                   'input_19_f2_layer': w2v_19_f2_train,
                   'input_20_f1_layer': w2v_20_f1_train,
                   'input_20_f2_layer': w2v_20_f2_train,
                   'input_21_f1_layer': w2v_21_f1_train,
                   'input_21_f2_layer': w2v_21_f2_train,
                   'input_22_f1_layer': w2v_22_f1_train,
                   'input_22_f2_layer': w2v_22_f2_train,
                   'input_23_f1_layer': w2v_23_f1_train,
                   'input_23_f2_layer': w2v_23_f2_train,
                   'input_24_f1_layer': w2v_24_f1_train,
                   'input_24_f2_layer': w2v_24_f2_train,
                   'input_25_f1_layer': w2v_25_f1_train,
                   'input_25_f2_layer': w2v_25_f2_train,
                   'input_26_f1_layer': w2v_26_f1_train,
                   'input_26_f2_layer': w2v_26_f2_train,
                   'input_27_f1_layer': w2v_27_f1_train,
                   'input_27_f2_layer': w2v_27_f2_train,
                   'input_28_f1_layer': w2v_28_f1_train,
                   'input_28_f2_layer': w2v_28_f2_train,
                   'input_29_f1_layer': w2v_29_f1_train,
                   'input_29_f2_layer': w2v_29_f2_train,
                   'input_30_f1_layer': w2v_30_f1_train,
                   'input_30_f2_layer': w2v_30_f2_train,
                   'input_31_f1_layer': w2v_31_f1_train,
                   'input_31_f2_layer': w2v_31_f2_train,
                   'input_32_f1_layer': w2v_32_f1_train,
                   'input_32_f2_layer': w2v_32_f2_train,
                   'input_33_f1_layer': w2v_33_f1_train,
                   'input_33_f2_layer': w2v_33_f2_train,
                   'input_34_f1_layer': w2v_34_f1_train,
                   'input_34_f2_layer': w2v_34_f2_train,
                   'input_35_f1_layer': w2v_35_f1_train,
                   'input_35_f2_layer': w2v_35_f2_train}
    input_valid = input_train.copy()
    input_test = {'input_1_f1_layer': w2v_1_f1_test,
                  'input_1_f2_layer': w2v_1_f2_test,
                  'input_2_f1_layer': w2v_2_f1_test,
                  'input_2_f2_layer': w2v_2_f2_test,
                  'input_3_f1_layer': w2v_3_f1_test,
                  'input_3_f2_layer': w2v_3_f2_test,
                  'input_4_f1_layer': w2v_4_f1_test,
                  'input_4_f2_layer': w2v_4_f2_test,
                  'input_5_f1_layer': w2v_5_f1_test,
                  'input_5_f2_layer': w2v_5_f2_test,
                  'input_6_f1_layer': w2v_6_f1_test,
                  'input_6_f2_layer': w2v_6_f2_test,
                  'input_7_f1_layer': w2v_7_f1_test,
                  'input_7_f2_layer': w2v_7_f2_test,
                  'input_8_f1_layer': w2v_8_f1_test,
                  'input_8_f2_layer': w2v_8_f2_test,
                  'input_9_f1_layer': w2v_9_f1_test,
                  'input_9_f2_layer': w2v_9_f2_test,
                  'input_10_f1_layer': w2v_10_f1_test,
                  'input_10_f2_layer': w2v_10_f2_test,
                  'input_11_f1_layer': w2v_11_f1_test,
                  'input_11_f2_layer': w2v_11_f2_test,
                  'input_12_f1_layer': w2v_12_f1_test,
                  'input_12_f2_layer': w2v_12_f2_test,
                  'input_13_f1_layer': w2v_13_f1_test,
                  'input_13_f2_layer': w2v_13_f2_test,
                  'input_14_f1_layer': w2v_14_f1_test,
                  'input_14_f2_layer': w2v_14_f2_test,
                  'input_15_f1_layer': w2v_15_f1_test,
                  'input_15_f2_layer': w2v_15_f2_test,
                  'input_16_f1_layer': w2v_16_f1_test,
                  'input_16_f2_layer': w2v_16_f2_test,
                  'input_17_f1_layer': w2v_17_f1_test,
                  'input_17_f2_layer': w2v_17_f2_test,
                  'input_18_f1_layer': w2v_18_f1_test,
                  'input_18_f2_layer': w2v_18_f2_test,
                  'input_19_f1_layer': w2v_19_f1_test,
                  'input_19_f2_layer': w2v_19_f2_test,
                  'input_20_f1_layer': w2v_20_f1_test,
                  'input_20_f2_layer': w2v_20_f2_test,
                  'input_21_f1_layer': w2v_21_f1_test,
                  'input_21_f2_layer': w2v_21_f2_test,
                  'input_22_f1_layer': w2v_22_f1_test,
                  'input_22_f2_layer': w2v_22_f2_test,
                  'input_23_f1_layer': w2v_23_f1_test,
                  'input_23_f2_layer': w2v_23_f2_test,
                  'input_24_f1_layer': w2v_24_f1_test,
                  'input_24_f2_layer': w2v_24_f2_test,
                  'input_25_f1_layer': w2v_25_f1_test,
                  'input_25_f2_layer': w2v_25_f2_test,
                  'input_26_f1_layer': w2v_26_f1_test,
                  'input_26_f2_layer': w2v_26_f2_test,
                  'input_27_f1_layer': w2v_27_f1_test,
                  'input_27_f2_layer': w2v_27_f2_test,
                  'input_28_f1_layer': w2v_28_f1_test,
                  'input_28_f2_layer': w2v_28_f2_test,
                  'input_29_f1_layer': w2v_29_f1_test,
                  'input_29_f2_layer': w2v_29_f2_test,
                  'input_30_f1_layer': w2v_30_f1_test,
                  'input_30_f2_layer': w2v_30_f2_test,
                  'input_31_f1_layer': w2v_31_f1_test,
                  'input_31_f2_layer': w2v_31_f2_test,
                  'input_32_f1_layer': w2v_32_f1_test,
                  'input_32_f2_layer': w2v_32_f2_test,
                  'input_33_f1_layer': w2v_33_f1_test,
                  'input_33_f2_layer': w2v_33_f2_test,
                  'input_34_f1_layer': w2v_34_f1_test,
                  'input_34_f2_layer': w2v_34_f2_test,
                  'input_35_f1_layer': w2v_35_f1_test,
                  'input_35_f2_layer': w2v_35_f2_test}
    return input_train, input_valid, input_test


# ## 本脚本仅保留第10折的运行记录

# In[9]:


count = 0
bs = 2048
gc.collect()
deepfm_data = deepfm_data.reset_index(drop=True).reset_index()
idtrain = list(deepfm_data[deepfm_data["raw_index"] < 6000_0000]['index'])
idtest = list(deepfm_data[deepfm_data["raw_index"] >= 6000_0000]['index'])
del deepfm_data
gc.collect()
for i, (train_index, test_index) in enumerate(skf.split(trainvalid, trainvalid['label'])):
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
    count += 1
    if count != 10:
        continue
    model = M(emb_mtx_f1_1, emb_mtx_f1_2, emb_mtx_f1_3, emb_mtx_f1_4, emb_mtx_f1_5,
              emb_mtx_f1_6, emb_mtx_f1_7, emb_mtx_f1_8, emb_mtx_f1_9, emb_mtx_f1_10,
              emb_mtx_f1_11, emb_mtx_f1_12, emb_mtx_f1_13, emb_mtx_f1_14, emb_mtx_f1_15,
              emb_mtx_f1_16, emb_mtx_f1_17, emb_mtx_f1_18, emb_mtx_f1_19, emb_mtx_f1_20,
              emb_mtx_f1_21, emb_mtx_f1_22, emb_mtx_f1_23, emb_mtx_f1_24, emb_mtx_f1_25,
              emb_mtx_f1_26, emb_mtx_f1_27, emb_mtx_f1_28, emb_mtx_f1_29, emb_mtx_f1_30,
              emb_mtx_f1_31, emb_mtx_f1_32, emb_mtx_f1_33, emb_mtx_f1_34, emb_mtx_f1_35,
              #!###
              emb_mtx_f2_1, emb_mtx_f2_2, emb_mtx_f2_3, emb_mtx_f2_4, emb_mtx_f2_5,
              emb_mtx_f2_6, emb_mtx_f2_7, emb_mtx_f2_8, emb_mtx_f2_9, emb_mtx_f2_10,
              emb_mtx_f2_11, emb_mtx_f2_12, emb_mtx_f2_13, emb_mtx_f2_14, emb_mtx_f2_15,
              emb_mtx_f2_16, emb_mtx_f2_17, emb_mtx_f2_18, emb_mtx_f2_19, emb_mtx_f2_20,
              emb_mtx_f2_21, emb_mtx_f2_22, emb_mtx_f2_23, emb_mtx_f2_24, emb_mtx_f2_25,
              emb_mtx_f2_26, emb_mtx_f2_27, emb_mtx_f2_28, emb_mtx_f2_29, emb_mtx_f2_30,
              emb_mtx_f2_31, emb_mtx_f2_32, emb_mtx_f2_33, emb_mtx_f2_34, emb_mtx_f2_35,
              #!###
              linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary',
              dnn_hidden_units=(256, 256, 256))
    print(' model compile finish ……')
    # 模型输入
    input_train, input_valid, input_test = get_input()
    # 训练集
    deepfm_train_input = {
        name: deepfm_train[name].iloc[train_index] for name in feature_names}
    for key in input_train.keys():
        input_train[key] = input_train[key][train_index]
    input_train.update(deepfm_train_input)
    for var in id_list_dict_emb_all.keys():
        input_train[var+'_seq_layer'] = id_list_dict_emb_all[var][0][idtrain][train_index]
    print('train input built!')
    # 验证集
    deepfm_valid_input = {
        name: deepfm_train[name].iloc[test_index] for name in feature_names}
    for key in input_valid.keys():
        input_valid[key] = input_valid[key][test_index]
    input_valid.update(deepfm_valid_input)
    for var in id_list_dict_emb_all.keys():
        input_valid[var+'_seq_layer'] = id_list_dict_emb_all[var][0][idtrain][test_index]
    print('valid input built!')
    callbacks = get_callbacks(i)
    hist = model.fit(input_train, label.values[train_index],
                     epochs=25, verbose=1, callbacks=callbacks,
                     batch_size=bs, validation_data=(input_valid, label.values[test_index]))
    print(hist.history)
    del deepfm_train_input
    gc.collect()
    # 测试集
    deepfm_test_input = {name: deepfm_test[name] for name in feature_names}
    input_test.update(deepfm_test_input)
    for var in id_list_dict_emb_all.keys():
        input_test[var+'_seq_layer'] = id_list_dict_emb_all[var][0][idtest]
    print('test input built!')
    # 预测
    y_pre = model.predict(input_test, verbose=2, batch_size=bs)
    np.save(f'../zlh_cache/model0929_addseq_f{i}.npy', y_pre)
    res['probability'] = res['probability'].values.reshape(
        -1, 1)+y_pre.reshape(-1, 1)/10
    # 预测验证集
    y_valid = model.predict(input_valid, verbose=2, batch_size=bs)
    np.save(f'../zlh_cache/model0929_addseq_valid_f{i}.npy', y_valid)
    trainvalid['probability'].iloc[test_index] = y_valid.ravel()
    del deepfm_valid_input, input_valid, deepfm_test_input, input_test
    gc.collect()
# # 整体保存作为oof后续参与stacking
df_save = pd.concat([trainvalid, res], axis=0)
df_save.to_pickle('model0929_addseq_10_folds_oof.pkl')
# subs
res = res[['id', 'probability']]
res = res.sort_values('id')
res.to_csv('submission_nn_zlh_10folds_0929_addseq.csv', index=False)
