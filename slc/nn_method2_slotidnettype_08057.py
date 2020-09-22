#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import Pool
import random
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input
from deepctr.layers.interaction import FM
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.utils import multi_gpu_model
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
import tensorflow.keras.backend as K
import tensorflow as tf
import joblib
from base.trans_layer import PositionEncoding
from base.trans_layer import MultiHeadAttention, PositionWiseFeedForward
from base.trans_layer import Add, LayerNormalization
from tqdm import tqdm
import os
import gc
from base import Cache
from itertools import chain
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


SEED = 999
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
gpus = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)


# ## 加载denses seqs embs

# In[2]:


print('load data start!')
# data = Cache.reload_cache('CACHE_data_step_2_feature_0917_r5.pkl')# 基础特征 0.797+
# del data['level_0']
# del data['communication_onlinerate']
# gc.collect()
n_slotid_nettype = np.load(
    './cached_data/EMB_slot_id_net_type_embedding_matrix.npy')
dense_feature_size = n_slotid_nettype.shape[-1]
gc.collect()
# window特征+2k

last_seq_list = ['creat_type_cd', 'tags',
                 'spread_app_id', 'task_id', 'adv_id', 'label']
user_fe_list = ['age', 'city_rank', 'career', 'gender',
                'city', 'device_name', 'residence', 'emui_dev']
item_fe_list = ['task_id', 'adv_id', 'adv_prim_id', 'tags', 'spread_app_id']
cross_emb_dict = {}  # 成对做拼接+slotnettype
for i, vari in enumerate(user_fe_list):
    for j, varj in enumerate(item_fe_list):
        if j > i:
            # 拼接emb
            df1 = Cache.reload_cache(
                f'CACHE_EMB_TARGET_DICT_{vari}__{varj}_w2v.pkl')
            df2 = Cache.reload_cache(
                f'CACHE_EMB_TARGET_DICT_{varj}__{vari}_w2v.pkl')
            embvari = {}
            # 都转int key ,拼接
            for key, value in df1['key'].items():
                embvari[key] = np.hstack([value, df2['value'][str(key)]])
            embvarj = {}
            # 都转int key ,拼接
            for key, value in df2['key'].items():
                embvarj[key] = np.hstack([value, df1['value'][str(key)]])
            cross_emb_dict[vari+'__'+varj] = (embvari, embvarj)
print('load data finish!')


# ## 处理做交叉相似度计算的列 生成索引

# In[3]:


# cols_to_emb = user_fe_list+item_fe_list
# train = data.query('label==label').copy()
# test = data.query('label!=label').copy()
# for var in cols_to_emb:
#     data[var+'_cemb'] = data[var].copy()
#     set1 = set(train[var]).intersection(set(test[var]))
#     print(var,' rare size: ',len(set1),data.loc[~data[var].isin(set1)].shape[0])
#     if len(set1)>0:
#         data.loc[~data[var].isin(set1),var+'_cemb']=-1
# del train,test,set1
# gc.collect()
# # 太慢，算完存过了
# Cache.cache_data(data, nm_marker='data_step_2_feature_0917_r5_crossembcol')
data = Cache.reload_cache('CACHE_data_step_2_feature_0917_r5_crossembcol.pkl')
# 重置index唯一值
del data['index']
data = data.reset_index(drop=True).reset_index()


# ## 生成预训练的cross_emb 列matrix

# In[4]:


def get_emb_matrix(col):
    """
    inputs:    
    col 需要做成预训练emb_matrix的列

    cross_emb_dict 结构：
    （embvari,embvarj）
    embvari:{key:word in dataframe,value:embvec} 就是字典

    data[col].unique() 需要转化的字典 不在原字典里的给-1 在的按大小顺序从1开始排

    得出id_list_dict + emb_matrix
    """
    vari, varj = col.split('__')
    key_to_represent_rare = -1
    words_vari = list(cross_emb_dict[col][0].keys())
    words_varj = list(cross_emb_dict[col][1].keys())
    emb_size_vari = cross_emb_dict[col][0][words_vari[0]].shape[0]
    emb_size_varj = cross_emb_dict[col][1][words_varj[0]].shape[0]
    voc_size_vari = len(words_vari)
    voc_size_varj = len(words_varj)
    list_df_vari = list(data[vari].unique())
    list_df_varj = list(data[varj].unique())
    # emb 中必须要有'-1' 作为index 0
    if -1 not in cross_emb_dict[col][0].keys():
        #  emb中无-1 为全词表数据！需要自行计算均值emb vec
        # 为embi 添加一个embedding
        # 这些词的vector求均值
        vector_low_frequency_words = np.zeros((emb_size_vari,))
        for w in words_vari:
            vector_low_frequency_words += cross_emb_dict[col][0][w]
        vector_low_frequency_words = vector_low_frequency_words / voc_size_vari
        # emb添加一个key value
        cross_emb_dict[col][0][key_to_represent_rare] = vector_low_frequency_words
        voc_size_vari += 1
        # print(f'{col} file has no key_to_represent_rare add low frequency words and fill vector as:', vector_low_frequency_words)
    if -1 not in cross_emb_dict[col][1].keys():
        #  emb中无-1 为全词表数据！需要自行计算均值emb vec
        # 为embi 添加一个embedding
        # 这些词的vector求均值
        vector_low_frequency_words = np.zeros((emb_size_varj,))
        for w in words_varj:
            vector_low_frequency_words += cross_emb_dict[col][1][w]
        vector_low_frequency_words = vector_low_frequency_words / voc_size_vari
        # emb添加一个key value
        cross_emb_dict[col][1][key_to_represent_rare] = vector_low_frequency_words
        voc_size_varj += 1
        # print(f'{col} file has no key_to_represent_rare add low frequency words and fill vector as:', vector_low_frequency_words)

    # 根据list_df_vari 生成emb matrix
    emb_matrix_vari = np.zeros((voc_size_vari + 1, emb_size_vari))  # 0是padding
    emb_matrix_varj = np.zeros((voc_size_varj + 1, emb_size_varj))  # 0是padding
    key2index_vari = {}  # 要对data[vari]做mapping
    key2index_varj = {}  # 要对data[varj]做mapping
    indexi = 2  # 1设为-1
    for k, idx in enumerate(list_df_vari):
        if idx in cross_emb_dict[col][0].keys():
            # 出现过
            emb_matrix_vari[indexi, :] = cross_emb_dict[col][0][idx]
            key2index_vari[idx] = indexi
            indexi += 1
        else:
            # 没出现过认为是-1
            key2index_vari[idx] = 1
    indexi = 2  # 1设为-1
    for k, idx in enumerate(list_df_varj):
        if idx in cross_emb_dict[col][1].keys():
            # 出现过
            emb_matrix_varj[indexi, :] = cross_emb_dict[col][1][idx]
            key2index_varj[idx] = indexi
            indexi += 1
        else:
            # 没出现过认为是-1
            key2index_varj[idx] = 1
    emb_matrix_vari = np.float32(emb_matrix_vari)
    emb_matrix_varj = np.float32(emb_matrix_varj)
    # 制作输入
    id_list_dict_vari = []  # input vari
    id_list_dict_varj = []  # input varj
    for valuei in tqdm(list(data[vari])):
        id_list_dict_vari.append(np.array([key2index_vari[valuei]]))
    for valuej in tqdm(list(data[varj])):
        id_list_dict_varj.append(np.array([key2index_varj[valuej]]))
    Cache.cache_data([(id_list_dict_vari, emb_matrix_vari), (id_list_dict_varj,
                                                             emb_matrix_varj)],                     nm_marker=f'CROSSEMB__{col}')


cross_emb_list = []
for i, vari in enumerate(user_fe_list):
    for j, varj in enumerate(item_fe_list):
        if j > i:
            cross_emb_list.append(vari+'__'+varj)
# 做一遍就可以了
# with Pool(10) as p:
#     p.map(get_emb_matrix, cross_emb_list)
print('Pool finish！')
id_list_dict_cross_emb_all = {}
dict_cross_emb_all = {}
for item in cross_emb_list:
    cross_emb = Cache.reload_cache(f'CACHE_CROSSEMB__{item}.pkl')
    id_list_dict_cross_emb_all[item+f'_i'] = cross_emb[0][0]
    id_list_dict_cross_emb_all[item+f'_j'] = cross_emb[1][0]
    dict_cross_emb_all[item+f'_i'] = cross_emb[0][1]
    dict_cross_emb_all[item+f'_j'] = cross_emb[1][1]
del cross_emb
gc.collect()
print('cross emb load finish！')


# ## 生成预训练的id_list_dict & matrix

# In[5]:


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
    id_list_dict_all = Cache.reload_cache(f'CACHE_EMB_INPUTSEQ_{col}.pkl')
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


with Pool(3) as p:
    res = p.map(get_emb_matrix, last_seq_list)
id_list_dict_emb_all = {}
for item in res:
    id_list_dict_emb_all.update(item)
del res, item
gc.collect()


# In[12]:


GlobalSeqLength = 40
base_inputdim_dict = {}
for var in id_list_dict_emb_all.keys():
    base_inputdim_dict[var] = id_list_dict_emb_all[var][1].shape[0]
base_embdim_dict = {'creat_type_cd': 32, 'tags': 32,
                    'spread_app_id': 32, 'task_id': 32, 'adv_id': 32, 'label': 32}
conv1d_info_dict = {'creat_type_cd': 8, 'tags': 8,
                    'spread_app_id': 8, 'task_id': 16, 'adv_id': 16, 'label': 8}
TRAINABLE_DICT = {'creat_type_cd': False, 'tags': False,
                  'spread_app_id': False, 'task_id': False, 'adv_id': False, 'label': False}
arr_name_list = list(id_list_dict_emb_all.keys())  # 过去行为序列
cross_arr_name_list = list(id_list_dict_cross_emb_all.keys())  # cross col emb


def get_seq_input_layers(cols):
    print("Prepare input layer:", cols)
    inputs_dict = {}
    for col in cols:
        inputs_dict[col] = tf.keras.Input(shape=(GlobalSeqLength, ),
                                          dtype="int32",
                                          name=col+'_seq_layer')
    return inputs_dict


def get_cross_seq_input_layers(cols):
    print("Prepare input layer:", cols)
    inputs_dict = {}
    for col in cols:
        inputs_dict[col] = tf.keras.Input(shape=(1, ),
                                          dtype="int32",
                                          name=col)
    return inputs_dict


def get_input_feature_layer(name=None, feature_shape=dense_feature_size, dtype="float32"):
    input_layer = tf.keras.Input(
        shape=(feature_shape,), dtype=dtype, name=name)
    return input_layer


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


def cross_net(inputsi, inputj, slotid_nettype, hidden_unit=4):
    x = tf.keras.layers.concatenate([inputsi, inputj, slotid_nettype])
    x = tf.keras.layers.Dense(hidden_unit, activation='relu',)(x)
    return x


def create_model(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary'):

    K.clear_session()
#!################################################################################################################
    inputs_all = [get_input_feature_layer(
        name='slotid_nettype', feature_shape=dense_feature_size)]
    # slotid_nettype
    layer_slotid_nettype = inputs_all[0]
    layer_slotid_nettype = K.expand_dims(layer_slotid_nettype, 1)
#!################################################################################################################
    seq_inputs_dict = get_cross_seq_input_layers(cols=cross_arr_name_list)
    inputs_all = inputs_all + list(seq_inputs_dict.values())  # 输入层list 做交叉

    cross_emb_out = []
    last_col = ''
    for index, col in enumerate(cross_arr_name_list):
        #         print(col, 'get embedding!')
        emb_layer = get_emb_layer(
            col, trainable=False, emb_matrix=dict_cross_emb_all[col])
        x = emb_layer(inputs_all[1+index])
        if col.split('_')[-1] == 'i':
            cross_user_item_i = x
            last_col = col
            continue
        else:
            print(f'crossing net add {last_col} and {col}')
            cross_emb_out.append(
                cross_net(cross_user_item_i, x, layer_slotid_nettype, hidden_unit=4))
    cross_emb_out = tf.keras.layers.concatenate(cross_emb_out)
    cross_emb_out = tf.squeeze(cross_emb_out, [1])
#!################################################################################################################
    seq_inputs_dict = get_seq_input_layers(cols=arr_name_list)
    inputs_all = inputs_all+list(seq_inputs_dict.values())  # 输入层list
    masks = tf.equal(seq_inputs_dict['task_id'], 0)
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

    mix = concatenate([cross_emb_out, trans_output,
                       dnn_input], axis=-1)  # !#mix

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(mix)

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

#!################################################################################################################

    model = Model(inputs=inputs_all+[features],
                  outputs=[output])
    print(model.summary())
    return model


# In[7]:


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


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, id_list_dict, df, feature_names, list_IDs, idtrain, batch_size=512, shuffle=True):
        self.id_list_dict = id_list_dict  # seq data to split
        self.df = df  # data_to split
        self.feature_names = feature_names
        self.batch_size = batch_size  # bs actually use
        self.list_IDs = list_IDs  # index range(all samples should train)
        self.idtrain = idtrain  # train data from id_list_dict
        self.shuffle = shuffle  # id true shuffle samples
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(list_IDs_temp)  # bs
        # X{name:df[name] for name in feature_names} y label
        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        X = {}
        for key in self.id_list_dict.keys():
            # id_list_dict[key][0]为seq
            X[key+'_seq_layer'] = self.id_list_dict[key][0][self.idtrain][list_IDs_temp]
        for key in self.feature_names:
            X[key] = self.df[key].iloc[list_IDs_temp]
        Y = self.df['label'].iloc[list_IDs_temp]
        return X, Y


# In[8]:


# 输入列
sparse_features = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career', 'gender', 'net_type', 'residence', 'emui_dev',
                   'indu_name', 'cmr_0', 'cmr_1', 'cmr_2', 'cmr_3', 'cmr_4', 'cmr_5', 'cmr_6', 'cmr_7', 'cmr_8', 'cmr_9', 'cmr_10', 'cmr_11', 'cmr_12', 'cmr_13', 'cmr_14', 'cmr_15', 'cmr_16', 'cmr_17', 'cmr_18', 'cmr_19', 'cmr_20', 'cmr_21', 'cmr_22', 'cmr_23', 'age', 'city_rank']
dense_features = [i for i in data.columns if i not in sparse_features +
                  ['index', 'id', 'uid', 'level_0', 'pt_d', 'label'] and i.find('_cemb') == -1]
# dense_features = dense_features+add_fe
print('sparse_features:')
print(sparse_features)
print('dense_features:')
print(dense_features)

# 特征处理
# Label Encoding for sparse features,and do simple Transformation for dense features
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
for feat in tqdm(dense_features):
    if feat.find('EMB') == -1:
        mms = MinMaxScaler(feature_range=(0, 1))
        if data[feat].max() > 2**32-1:
            data[feat] = data[feat].replace(np.Inf, 2**16-1)
        data[feat] = mms.fit_transform(
            data[feat].values.reshape(-1, 1))  # .astype(np.float32)
    if data[feat].isnull().sum() > 0:
        data[feat] = data[feat].fillna(data[feat].max())
droplist = []
for var in tqdm(sparse_features+dense_features):
    if data[var].nunique() < 2 or data[var].count() < 2:
        droplist.append(var)
for var in droplist:
    dense_features.remove(var)
    del data[var]
print('find droplist:', droplist)
gc.collect()
#!################################################################################################################
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=8)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                        for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print('feature_names finish!')


# In[13]:


# callback
lr_list = [0.001, 0.001, 0.001, 0.0005, 0.00025,
           0.000125, 6.25e-05, 3.125e-05, 2e-05, 2e-05, 2e-05]


def scheduler(epoch):
    if epoch < len(lr_list):
        return lr_list[epoch]
    else:
        return 2.5e-6


def get_callbacks(if_valid=True):
    '''
    :param count:
    :return:
    '''
    checkpoint_dir = 'models'
    checkpoint_prefix = os.path.join(
        checkpoint_dir, f"ckpt_zlhnn_model0920_m2slotidnettype_{if_valid}")

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
        filename='./logs/model_zlhnn_model0920_m2slotidnettype.log', separator=",", append=True)

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
print(' model compile start ……')
try:
    del model
    gc.collect()
    K.clear_session()
except:
    pass
model = create_model(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                     dnn_hidden_units=(512, 256, 256), dnn_dropout=0.1, task='binary')
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=multi_category_focal_loss2(alpha=0.35),
    metrics=['AUC'])
print(' model compile finish ……')


# In[14]:


bs = 2048+512
count = 0
random_state = 1111
set1 = data.query('pt_d<8').copy()
set2 = data.query('pt_d==8').copy()
# del data
gc.collect()
idtrain = list(set1['index'])
idtest = list(set2['index'])
skf = StratifiedKFold(
    n_splits=10, random_state=random_state, shuffle=True)  # 抽90% 训练
for i, (train_index, test_index) in enumerate(skf.split(set1, set1['label'])):
    print("FOLD | ", count+1)
    print("###"*35)
    gc.collect()

    # 模型输入
    # 训练集
    online_train_model_input = {}
    online_train_model_input['slotid_nettype'] = n_slotid_nettype[idtrain][train_index]
    for var in id_list_dict_cross_emb_all.keys():
        online_train_model_input[var] = np.array(id_list_dict_cross_emb_all[var])[
            idtrain][train_index]
    for var in id_list_dict_emb_all.keys():
        online_train_model_input[var +
                                 '_seq_layer'] = id_list_dict_emb_all[var][0][idtrain][train_index]
    online_train_model_input.update(
        {name: set1[name].values[train_index] for name in tqdm(feature_names)})
    y_true_train = set1['label'].values[train_index]
    print('train input built!')
    # 验证集
    online_valid_model_input = {}
    online_valid_model_input['slotid_nettype'] = n_slotid_nettype[idtrain][test_index]
    for var in id_list_dict_cross_emb_all.keys():
        online_valid_model_input[var] = np.array(id_list_dict_cross_emb_all[var])[
            idtrain][test_index]
    for var in id_list_dict_emb_all.keys():
        online_valid_model_input[var +
                                 '_seq_layer'] = id_list_dict_emb_all[var][0][idtrain][test_index]
    online_valid_model_input.update(
        {name: set1[name].values[test_index] for name in tqdm(feature_names)})
    y_true_valid = set1['label'].values[test_index]
    print('valid input built!')
    # 测试集
    online_test_model_input = {}
    online_test_model_input['slotid_nettype'] = n_slotid_nettype[idtest]
    for var in id_list_dict_cross_emb_all.keys():
        online_test_model_input[var] = np.array(
            id_list_dict_cross_emb_all[var])[idtest]
    for var in id_list_dict_emb_all.keys():
        online_test_model_input[var +
                                '_seq_layer'] = id_list_dict_emb_all[var][0][idtest]
    online_test_model_input.update(
        {name: set2[name].values for name in tqdm(feature_names)})
    print('test input built!')
    callbacks = get_callbacks()
    hist = model.fit(online_train_model_input, y_true_train,
                     epochs=40,
                     batch_size=bs,
                     verbose=1,
                     callbacks=callbacks,
                     validation_data=(online_valid_model_input, y_true_valid))
    print(hist.history)
    if count == 0:
        break
# 预测
y_pre = model.predict(
    online_test_model_input, verbose=1, batch_size=1024)
res = set2[['id']]
res['probability'] = y_pre
res = res.sort_values('id')
res.to_csv('./subs/submission_nn_0920_m2slotidnettype.csv', index=False)


# In[15]:


# 预测
y_pre = model.predict(
    online_test_model_input, verbose=1, batch_size=1024)
res = set2[['id']]
res['probability'] = y_pre
res = res.sort_values('id')
res.to_csv('./subs/submission_nn_0920_m2slotidnettype.csv', index=False)


# ## 线下0.82641 线上0.800218

# ## 15轮2048最好！

# ## 人工早停后线上 0.805725

# In[ ]:


# In[ ]:
