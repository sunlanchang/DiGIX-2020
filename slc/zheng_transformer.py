# -*- encoding: utf-8 -*-
'''
@File      :   train_with_keras_multi_gpu.py
@Time      :   2020/06/27 23:40:48
@Author    :   zhangqibot 
@Version   :   0.1
@Contact   :   hi@zhangqibot.com
@Desc      :   tensorflow1.10 + keras 2.2.4 多卡
'''

import gc
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from base import Cache
from base.attention import Attention
import keras.backend as K
from base import reduce_mem
from base import show_all_feas
from collections import OrderedDict
from base import rm_feas, get_cur_dt_str
from keras.layers import CuDNNLSTM, CuDNNGRU
import keras
from keras.utils import multi_gpu_model
import random
import argparse
import pickle
from base.emb import EmbBatchLoader
from base.emb import get_embedding_tool
import sys
import datetime
print(sys.path)
seed = 100
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ["PYTHONHASHSEED"] = str(seed)

print(tf.__version__)

# tm_now = get_cur_dt_str()
# CUR_FOLD = 0
##############################################################################
# 传参。例如：
# /root/miniconda3/envs/TF_2.2/bin/python -u run_by_fold.py --fold 4 --tm_now ${tm_now}
# 如果跑五折，--fold 如果不在 [0,1,2,3,4] 中，则全部五折会被一起训练。
##############################################################################

parser = argparse.ArgumentParser(description="kfold:0,1,2,3,4...")
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--tm_now', type=str, default="19920911")
args = parser.parse_args()
CUR_FOLD = args.fold
tm_now = args.tm_now.replace(":", "").replace("-", "")
# CUR_FOLD = -1
# tm_now = "202007141359"

print("###" * 35)
print("@@@tm_now:", tm_now)
print("@@@CUR_FOLD to train:", CUR_FOLD)
print("###" * 35)

##############################################################################
# 定义一些参数：
##############################################################################

NUM_WORKERS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # allocate dynamically
LABEL_CLASS = {'age': 10, 'gender': 2, 'age_gender': 20}
CUR_LABEL = 'age_gender'  # 'age'  # label
NUM_CLASSES = LABEL_CLASS[CUR_LABEL]
N_FOLDS = 5
BATCH_SIZE = 512 * NUM_WORKERS
VERBOSE = 2
EPOCHS = 50
USE_TRAINED_CKPT = False  # 是否用ckpt做预测
CKPT_BASE_DIR = './'
USE_SEQ_LENGTH = 150  # <= 最大序列长度
INPUT_DATA_BASE_DIR = "../cached_data"  # "cached_data/stage1_inputs/": 初赛数据
AUTHOR = "AZ"  # 用的谁的输入序列 需要标记一下 方便ensemble
YOUR_MARKER = "keras_seq_150_lstm_transformer_local_base"

# 【断点训练】：如果是断点训练，必须设置 NEW_TRAIN = False
# 且设置 TRAIN_MARKER 为上次训练的文件夹即可；
# 同时 FOLD2SKIP表示跳过之前的训练，如FOLD2SKIP=0，表示跳过第一折；FOLD2SKIP=1表示跳过前两折...
# 【从头训练】：如果是全新的训练 只需要设置 NEW_TRAIN = True

# # 【断点训练】
# TRAIN_MARKER = "0625_2022_age_gender_M4938_semi_20CLS_3V100"  # 断点训练
# FOLD2SKIP = 0  # if count <= FOLD2SKIP: continue  # 断点训练
# # if count <= FOLD2SKIP:
# #     print("Skip...")
# #     continue

# # 【从头训练】
# if NEW_TRAIN:
#     TRAIN_MARKER = None
#     FOLD2SKIP = 999

# if TRAIN_MARKER is None:
TRAIN_MARKER = f"{tm_now}_{CUR_LABEL}_{YOUR_MARKER}_{AUTHOR}"  # Important!

# 去掉 product_id
EMB_keys2do = [
    'creative_id', 'product_id', 'ad_id', 'advertiser_id', 'industry', 'product_category', 'time'
]

##############################################################################
os.makedirs(
    f"/home/tione/notebook/05_RESULT/META/{TRAIN_MARKER}", exist_ok=True)
os.makedirs("/home/tione/notebook/05_RESULT/SUB", exist_ok=True)

print("###" * 35)
print(f"tm_now: {tm_now}")
print("TRAIN_MARKER: ", TRAIN_MARKER)
print("CUR_LABEL: ", CUR_LABEL)
print("EPOCHS: ", EPOCHS)
print("NUM_WORKERS: ", NUM_WORKERS)
print("Cards to use:", os.environ["CUDA_VISIBLE_DEVICES"])
print("BATCH_SIZE: ", BATCH_SIZE)
print("EMB_keys2do: ", EMB_keys2do)
print("NUM_CLASSES: ", NUM_CLASSES)
print("USE_SEQ_LENGTH: ", USE_SEQ_LENGTH)
print("###" * 35)

##############################################################################

print("###" * 35)
print("@@@Load id_list_dict...")
print("###" * 35)
id_list_dict = Cache.reload_cache(file_nm='/home/tione/notebook/cached_data/CACHE_id_list_dict_150_normal.pkl',
                                  pure_nm=False)
gc.collect()
# id_list_dict 包含padding后的序列特征字典以及词表
# truncate:
if USE_SEQ_LENGTH < 150:
    for col in EMB_keys2do:
        id_list_dict[col + "_list"]['id_list'] = id_list_dict[
            col + "_list"]['id_list'][:, -USE_SEQ_LENGTH:]

SEQ_LENTH_DICT = {}  # 存放每个序列截断长度的字典 一般都是一样的，比如这里是 150

for key in EMB_keys2do:
    SEQ_LENTH_DICT[key] = id_list_dict[f'{key}_list']['id_list'].shape[-1]

if len(set(SEQ_LENTH_DICT.values())) == 1:
    GlobalSeqLength = SEQ_LENTH_DICT[key]
    print("GlobalSeqLength:", GlobalSeqLength)
else:
    print(
        "GlobalSeqLength is Not Unique!!! If you are sure, comment the line after to avoid exception."
    )

##############################################################################
print("###" * 35)
print("@@@Load datalabel...")
print("###" * 35)
# label
datalabel = pd.read_hdf(
    '/home/tione/notebook/cached_data/datalabel_original_stage2.h5')

if datalabel['age'].min() == 1:
    datalabel['age'] = datalabel['age'] - 1
if datalabel['gender'].min() == 1:
    datalabel['gender'] = datalabel['gender'] - 1
assert datalabel['age'].min() == 0
assert datalabel['gender'].min() == 0
datalabel = datalabel[['user_id', 'gender', 'age']]
traindata = datalabel.loc[~datalabel['age'].isna()].reset_index(drop=True)
testdata = datalabel.loc[datalabel['age'].isna()].copy().reset_index(drop=True)
traindata['age'] = traindata['age'].astype(np.int8)
traindata['gender'] = traindata['gender'].astype(np.int8)
traindata['age_gender'] = traindata['gender'] * 10 + traindata['age']
# gender = 0, age => 0~9
# gender = 1, age+=10 => 10~19
print(f"traindata['age_gender'].unique(): {traindata['age_gender'].unique()}")
print(traindata.shape, testdata.shape)

input_dict_all = {}
for col in EMB_keys2do:
    input_dict_all[col] = id_list_dict[col + '_list']['id_list']
gc.collect()

all_uid_df = datalabel[['user_id']].copy()  # to save the model_prob
train_uid_df = traindata[['user_id']].copy()  # to save the oof_prob
all_uid_df.to_csv(f"/home/tione/notebook/05_RESULT/META/{TRAIN_MARKER}/SAVE_all_uid_df.csv",
                  index=False)
train_uid_df.to_csv(f"/home/tione/notebook/05_RESULT/META/{TRAIN_MARKER}/SAVE_train_uid_df.csv",
                    index=False)
del datalabel
gc.collect()

# #################################################################################
# init array to store oof and model prob
train_shape = traindata.shape[0]
test_shape = testdata.shape[0]
oof_pred = np.zeros((train_shape, NUM_CLASSES))
model_prob = np.zeros((train_shape + test_shape, NUM_CLASSES, N_FOLDS))
# #################################################################################

print("###" * 35)
print("@@@Load Embedding...")
print('ALL_EMB_COLS_TO_USE:', EMB_keys2do)
print("###" * 35)
# 定义最大emb_size
max_embs = {'creative_id': 2000, 'ad_id': 2000, 'advertiser_id': 2000, 'product_id': 2000, 'product_category': 2000,
            'industry': 2000, 'click_times': 600, 'time': 600}
# 定义emb 文件路径
path_list0 = '/home/tione/notebook/cached_data/'
path1 = '/home/tione/notebook/cached_data/emb1/'
path_list = ['/home/tione/notebook/cached_data/',
             #              '/home/tione/notebook/cached_data/'
             ]
# 定义随机抽几个emb
fix_num = 2
max_nums = {'creative_id': 6, 'ad_id': 5,
            'advertiser_id': 5, 'product_id': 5,
            'product_category': 5, 'industry': 5,
            'click_times': 1, 'time': 2}
# base emb
special_userlist = {'creative_id': [
    path_list0 + 'CACHE_EMB_DICT_AZ_sg_CONCAT_time_diff_60WINDOW_10EPOCH_user_id_creative_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3MINUS_30WINDOW_10EPOCH_user_id_creative_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3PLUS_30WINDOW_10EPOCH_user_id_creative_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_creative_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_cbow_CONCAT_product_category_60WINDOW_10EPOCH_user_id_creative_id.pkl',
    path1 + 'CACHE_EMB_DICT_ZQ_DW_RM_CNT1_PATH100_50WINDOW_10EPOCH_user_id_creative_id.pkl'
],
    'ad_id': [
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3MINUS_30WINDOW_10EPOCH_user_id_ad_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3PLUS_30WINDOW_10EPOCH_user_id_ad_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_ad_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_ORIGINALBASE_30WINDOW_5EPOCH_user_id_ad_id.pkl',
    path1 + 'CACHE_EMB_DICT_ZQ_DW_RM_CNT1_20WINDOW_10EPOCH_user_id_ad_id.pkl'
],
    'advertiser_id': [
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3MINUS_30WINDOW_10EPOCH_user_id_advertiser_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3PLUS_30WINDOW_10EPOCH_user_id_advertiser_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_advertiser_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_cbow_CONCAT_industry_60WINDOW_10EPOCH_user_id_advertiser_id.pkl',
    path1 + 'CACHE_EMB_DICT_ZQ_DW_RM_CNT1_20WINDOW_10EPOCH_user_id_advertiser_id.pkl'
],
    'product_id': [
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3MINUS_30WINDOW_10EPOCH_user_id_product_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3PLUS_30WINDOW_10EPOCH_user_id_product_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_product_id.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_ORIGINALBASE_30WINDOW_5EPOCH_user_id_product_id.pkl',
    path1 + 'CACHE_EMB_DICT_ZQ_DW_RM_CNT1_20WINDOW_10EPOCH_user_id_product_id.pkl'
],
    'product_category': [
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3MINUS_30WINDOW_10EPOCH_user_id_product_category.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3PLUS_30WINDOW_10EPOCH_user_id_product_category.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_product_category.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_ORIGINALBASE_30WINDOW_5EPOCH_user_id_product_category.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TXBASE_150WINDOW_10EPOCH_user_id_product_category.pkl'
],
    'industry': [
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3MINUS_30WINDOW_10EPOCH_user_id_industry.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TIMEDIFF3PLUS_30WINDOW_10EPOCH_user_id_industry.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_CLICK_TIMES_INCREASED_60WINDOW_10EPOCH_user_id_industry.pkl',
                 path_list0 + 'CACHE_EMB_DICT_AZ_cbow_CONCAT_industry_60WINDOW_10EPOCH_user_id_industry.pkl',
    path1 + 'CACHE_EMB_DICT_ZQ_DW_RM_CNT1_20WINDOW_10EPOCH_user_id_industry.pkl'
],
    'click_times': [],
    'time': [
    path_list0 + 'CACHE_EMB_DICT_AZ_timeemb_user_id_time.pkl',
    path_list0 + 'CACHE_EMB_DICT_AZ_CBOW_TXBASE_150WINDOW_10EPOCH_user_id_time.pkl'
]}

gt = get_embedding_tool(max_embs=max_embs, max_nums=max_nums, use_cols=EMB_keys2do, path_list=path_list,
                        spec_emb_dict=special_userlist)  # click_times不查找embedding
concated_emb_dict = gt.random_get_embedding_fun(id_list_dict)
gc.collect()

# #################################################################################

for k, v in concated_emb_dict.items():
    print('used emb info:')
    print(k, v.shape)

gc.collect()
#######################################################################


def Adding_Layer(tensor):
    x, y = tensor
    return x + y


class ScaledDotProductAttention(keras.layers.Layer):

    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs * masks * self._masking_num
        return outputs

    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [
                               tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    def call(self, inputs):
        if self._masking:
            assert len(
                inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values = inputs
        else:
            assert len(
                inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':
            queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':
            keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':
            values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        # if self._masking:
        #     scaled_matmul = self.mask(scaled_matmul, masks)  # Mask(opt.)
        #
        # if self._future:
        #     scaled_matmul = self.future_mask(scaled_matmul)

        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)

        outputs = K.batch_dot(out, values)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(keras.layers.Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=True, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = False
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        if self._masking:
            assert len(
                inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(
                inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(
            tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(
            tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(
            tf.split(values_linear, self._n_heads, axis=2), axis=0)

        if self._masking:
            att_inputs = [queries_multi_heads,
                          keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads,
                          keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization(keras.layers.Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFeedForward(keras.layers.Layer):

    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")
        self.weights_out = self.add_weight(shape=(self._inner_dim, self._model_dim),
                                           initializer='glorot_uniform', trainable=self._trainable, name="weights_out")
        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")
        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim

#######################################################################


def get_seq_input_layers(cols=EMB_keys2do):
    print("Prepare input layer:", cols)
    inputs_dict = OrderedDict()
    for col in cols:
        inputs_dict[col] = keras.Input(shape=(GlobalSeqLength, ),
                                       dtype="int32",
                                       name=col)
    return inputs_dict


def get_emb_layer(emb_matrix, seq_length=None, trainable=False):
    if seq_length is None:
        seq_length = GlobalSeqLength  # 如果未指定 seq_length 就用 GlobalSeqLength
    embedding_dim = emb_matrix.shape[-1]
    input_dim = emb_matrix.shape[0]

    emb_layer = keras.layers.Embedding(input_dim,
                                       embedding_dim,
                                       input_length=seq_length,
                                       weights=[emb_matrix],
                                       dtype="float32",
                                       trainable=trainable)
    return emb_layer


def trans_net(input, n_unit=512):
    input = keras.layers.Dropout(0.3)(input)
    encodings = keras.layers.Conv1D(filters=input.shape[-1].value, kernel_size=1, padding='same', activation='relu')(
        input)

    for i in range(1):
        # pre Norm
        encodings = LayerNormalization()(encodings)
        # Masked-Multi-head-Attention
        masked_attention_out = MultiHeadAttention(
            8, encodings.shape[-1].value // 8)([encodings, encodings, encodings])
        # Add
        masked_attention_out = keras.layers.Lambda(
            Adding_Layer)([masked_attention_out, encodings])
        # pre Norm
        masked_attention_out = LayerNormalization()(masked_attention_out)
        # Feed-Forward
        ff = PositionWiseFeedForward(encodings.shape[-1].value, n_unit)
        ff_out = ff(masked_attention_out)
        # Add
        ff_out = keras.layers.Lambda(Adding_Layer)(
            [ff_out, masked_attention_out])
        encodings = ff_out
    return encodings


def Expand_Dim_Layer(tensor):
    def expand_dim(tensor):
        return K.expand_dims(tensor, axis=-1)
    return keras.layers.Lambda(expand_dim)(tensor)


def Adding_Layer(tensor):
    x, y = tensor
    return x + y


def create_model(unit, all_emb_cols):

    ######################################################################################################
    # Input
    ######################################################################################################

    inputs_dict = get_seq_input_layers(cols=all_emb_cols)
    inputs_all = list(inputs_dict.values())

    ######################################################################################################
    # embedding
    ######################################################################################################

    train_able_dict = {'creative_id': False, 'ad_id': False, 'advertiser_id': False,
                       'product_id': False, 'industry': True, 'product_category': True, 'time': False, 'click_times': False}
    conv1d_info_dict = {'creative_id': 256, 'ad_id': 128, 'advertiser_id': 128, 'industry': 64, 'product_category': 64,
                        'product_id': 128, 'time': 64, 'click_times': -1}

    emb_layers_dict = {}

    for col, emb_matrix in concated_emb_dict.items():

        if col not in all_emb_cols:
            continue

        trainable = train_able_dict[col]
        print(col, 'has been lookup for emb!')
        emb_layers_dict[col] = get_emb_layer(emb_matrix,
                                             seq_length=None,
                                             trainable=trainable)

        emb_layers_dict[col] = emb_layers_dict[col](inputs_dict[col])
        if conv1d_info_dict[col] > 0:
            emb_layers_dict[col] = keras.layers.Conv1D(filters=conv1d_info_dict[col],
                                                       kernel_size=1,
                                                       padding='same',
                                                       activation='relu')(emb_layers_dict[col])

    x = keras.layers.concatenate(list(emb_layers_dict.values()))
    x8 = keras.layers.Conv1D(filters=256, kernel_size=1,
                             padding='same', activation='relu')(x)
    x = trans_net(x, n_unit=512)
    x = keras.layers.concatenate([x, x8])
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv1D(
        filters=x.shape[-1].value, kernel_size=1, padding='same', activation='relu')(x)
    lstm = keras.layers.Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    conv = keras.layers.Conv1D(
        filters=256, kernel_size=1, padding='same', activation='relu')(x)

    max_pool = keras.layers.GlobalMaxPooling1D()
    average_pool = keras.layers.GlobalAveragePooling1D()
    x9 = max_pool(lstm)
    x10 = average_pool(lstm)
    x11 = max_pool(conv)
    x12 = average_pool(conv)
    x = keras.layers.concatenate([x9, x10, x11, x12])
    concat_all = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512)(concat_all)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.PReLU()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.PReLU()(x)
    outputs_all = keras.layers.Dense(NUM_CLASSES,
                                     activation='softmax',
                                     name='age_gender')(x)
    model = keras.Model(inputs_all, outputs_all)
    print(model.summary())
    return model


def get_callbacks():

    earlystop_callback = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=0.0001,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        restore_best_weights=True,
    )

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_acc',
        factor=0.5,
        patience=1,
        min_delta=2e-4,
        min_lr=0.00001)  # new_lr = lr * factor.
    callbacks = [earlystop_callback, reduce_lr_callback]
    return callbacks


def get_input_by_index(idx_lst):
    input_dict_trn = {}
    for col in EMB_keys2do:
        input_dict_trn[col] = input_dict_all[col][idx_lst]
    return input_dict_trn


class TX_Evaluate(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.score = {}
        self.best = 0.

    def on_epoch_end(self, epoch, logs=None):
        age_score, gender_score = self._evaluate()
        score = age_score + gender_score
        self.score[epoch] = [age_score, gender_score, score]
        if score > self.best:
            self.best = score
            print(f"[!] epoch = {epoch + 1}, new best ACC = {score}")
        print(
            f'[!] epoch = {epoch + 1}, age_ACC = {age_score}, gender_ACC = {gender_score}, all_ACC = {score}, best ACC: {self.best}\n'
        )

    def _evaluate(self):
        prob = self.model.predict(self.X_val)  # (,20)
        prob_age = prob[:, :10] + prob[:, 10:]
        prob_gender = np.concatenate((prob[:, :10].sum(
            axis=1, keepdims=True), prob[:, 10:].sum(axis=1, keepdims=True)),
            axis=1)
        pred_age = np.argmax(prob_age, axis=1) + 1
        pred_gender = np.argmax(prob_gender, axis=1) + 1
        true_gender = np.array(
            list(map(lambda x: 1 if x >= 10 else 0, self.y_val)))
        true_age = (self.y_val - true_gender * 10)
        true_age += 1
        true_gender += 1
        age_acc = sum(true_age == pred_age) / len(true_age)
        gender_acc = sum(true_gender == pred_gender) / len(true_gender)
        return age_acc, gender_acc


# #################################################################################
# 五折开始啦~
# #################################################################################
# ID:
# all_uid_df
# train_uid_df
# #################################################################################


print("###" * 35)
print("Save id df ...")
print(f"{N_FOLDS} Fold Training Start...")
print(f"tm_now: {tm_now}")
print("###" * 35)

score_val = []
skf = StratifiedKFold(n_splits=N_FOLDS, random_state=1111, shuffle=True)
folds = list(skf.split(traindata, traindata[CUR_LABEL]))

with open(f"/home/tione/notebook/05_RESULT/META/{TRAIN_MARKER}/folds.pkl", 'wb') as file:
    pickle.dump(folds, file)

for count, (train_index, test_index) in enumerate(folds):
    print("###" * 35)
    print("FOLD | ", count)
    if (CUR_FOLD >= 0) and (CUR_FOLD < N_FOLDS):
        if count < CUR_FOLD:
            print("Skip...")
            continue
        if count > CUR_FOLD:
            print("Break...")
            break
    else:
        print(
            f"CUR_FOLD={CUR_FOLD}, Full 5 folds will be trained in this loop..."
        )
    print("###" * 35)
    input_dict_trn = get_input_by_index(train_index)
    y_true_trn = traindata[CUR_LABEL].values[train_index]
    input_dict_val = get_input_by_index(test_index)
    y_true_val = traindata[CUR_LABEL].values[test_index]
    try:
        del model
        gc.collect()
        K.clear_session()
    except:
        pass
    with tf.device("/cpu:0"):
        model = create_model(256,
                             all_emb_cols=EMB_keys2do)
        model = multi_gpu_model(model, NUM_WORKERS)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=5e-4),  # Adam
        loss='sparse_categorical_crossentropy',
        metrics=['acc'])

    if USE_TRAINED_CKPT:
        checkpoint_prefix = os.path.join(CKPT_BASE_DIR,
                                         f'ckpt_{TRAIN_MARKER}_FOLD_{count}')
        print("###" * 35)
        print("Use Trained Checkpoint...")
        print(checkpoint_prefix)
        print("###" * 35)
        model.load_weights(checkpoint_prefix)
        hist = None
    else:
        print("###" * 35)
        print("Train from Scratch...")
        print("###" * 35)
        callbacks = get_callbacks()
#         eval_callback = TX_Evaluate(X_val=input_dict_val, y_val=y_true_val)
#         callbacks += [eval_callback]
        hist = model.fit(input_dict_trn,
                         y_true_trn,
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         verbose=VERBOSE,
                         callbacks=callbacks,
                         validation_data=(input_dict_val, y_true_val))
    if hist is not None:
        print("###" * 35)
        print(hist.history)
        print("###" * 35)
        if "val_accuracy" in hist.history:
            key = "val_accuracy"
        elif "val_acc" in hist.history:
            key = "val_acc"
        else:
            key = "val_loss"
        print(f"history key: {key}")
        score_val.append(np.max(hist.history[key]))

    print("###" * 35)
    print(f"Make Prediction...Fold-{count}")
    pred_oof = model.predict(
        input_dict_val,
        batch_size=BATCH_SIZE,  # // NUM_WORKERS,
        verbose=VERBOSE)
    oof_pred[test_index, :] = pred_oof
    oof_pred = np.float32(oof_pred)
    np.save(
        f"/home/tione/notebook/05_RESULT/META/{TRAIN_MARKER}/SAVE_OOF", pred_oof
    )  # 还是存pred_oof，防止5折断开 # oof_pred 就每次覆盖存储 全量的 oof... 存 pred_oof 不方便。
    pred_all = model.predict(
        input_dict_all,
        batch_size=BATCH_SIZE,  # // NUM_WORKERS,
        verbose=VERBOSE)
    pred_all = np.float32(pred_all)
    model_prob[:, :, count] = pred_all
    np.save(f"/home/tione/notebook/05_RESULT/META/{TRAIN_MARKER}/SAVE_MODEL_PROB_FOLD{count}",
            pred_all)
    del pred_all
    gc.collect()
    print("Done Prediction!")
    print("###" * 35)

print(f"offline score mean: {score_val}")
print(f"offline score by folds: {np.mean(score_val)}")
print("All Done!")
