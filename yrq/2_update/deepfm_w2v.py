import numpy as np
import pandas as pd
from itertools import chain

from base import Cache

import os
from tqdm import tqdm

from trans_layers import Add, LayerNormalization
from trans_layers import MultiHeadAttention, PositionWiseFeedForward
from trans_layers import PositionEncoding

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import optimizers, layers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, GlobalMaxPooling1D, Flatten
from tensorflow.keras.backend import concatenate
from gensim.models import Word2Vec, KeyedVectors

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import  SparseFeat, DenseFeat, get_feature_names, build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, add_func, combined_dnn_input

def M(emb_mtx_f1_1,emb_mtx_f1_2,emb_mtx_f1_3,emb_mtx_f1_4,emb_mtx_f1_5,
    emb_mtx_f1_6,emb_mtx_f1_7,emb_mtx_f1_8,emb_mtx_f1_9,emb_mtx_f1_10,
    emb_mtx_f1_11,emb_mtx_f1_12,emb_mtx_f1_13,emb_mtx_f1_14,emb_mtx_f1_15,
    emb_mtx_f1_16,emb_mtx_f1_17,emb_mtx_f1_18,emb_mtx_f1_19,emb_mtx_f1_20,
    emb_mtx_f1_21,emb_mtx_f1_22,emb_mtx_f1_23,emb_mtx_f1_24,emb_mtx_f1_25,
    emb_mtx_f1_26,emb_mtx_f1_27,emb_mtx_f1_28,emb_mtx_f1_29,emb_mtx_f1_30,
    emb_mtx_f1_31,emb_mtx_f1_32,emb_mtx_f1_33,emb_mtx_f1_34,emb_mtx_f1_35,
    #!###
    emb_mtx_f2_1,emb_mtx_f2_2,emb_mtx_f2_3,emb_mtx_f2_4,emb_mtx_f2_5,
    emb_mtx_f2_6,emb_mtx_f2_7,emb_mtx_f2_8,emb_mtx_f2_9,emb_mtx_f2_10,
    emb_mtx_f2_11,emb_mtx_f2_12,emb_mtx_f2_13,emb_mtx_f2_14,emb_mtx_f2_15,
    emb_mtx_f2_16,emb_mtx_f2_17,emb_mtx_f2_18,emb_mtx_f2_19,emb_mtx_f2_20,
    emb_mtx_f2_21,emb_mtx_f2_22,emb_mtx_f2_23,emb_mtx_f2_24,emb_mtx_f2_25,
    emb_mtx_f2_26,emb_mtx_f2_27,emb_mtx_f2_28,emb_mtx_f2_29,emb_mtx_f2_30,
    emb_mtx_f2_31,emb_mtx_f2_32,emb_mtx_f2_33,emb_mtx_f2_34,emb_mtx_f2_35,
    #!###
    linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
    l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
    dnn_activation='elu', dnn_use_bn=False, task='binary'):

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

    concat_1 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_1 = Dense(2,activation='softmax',)(concat_1)
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

    concat_2 = concatenate([x1_2,x2_2] , axis=-1)

    output_f1_f2_2 = Dense(2,activation='softmax',)(concat_2)
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

    concat_3 = concatenate([x1_3,x2_3] , axis=-1)

    output_f1_f2_3 = Dense(2,activation='softmax',)(concat_3)
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

    concat_4 = concatenate([x1_4,x2_4] , axis=-1)

    output_f1_f2_4 = Dense(2,activation='softmax',)(concat_4)
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

    concat_5 = concatenate([x1_5,x2_5] , axis=-1)

    output_f1_f2_5 = Dense(2,activation='softmax',)(concat_5)
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

    concat_6 = concatenate([x1_6,x2_6] , axis=-1)

    output_f1_f2_6 = Dense(2,activation='softmax',)(concat_6)
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

    concat_7 = concatenate([x1_7,x2_7] , axis=-1)

    output_f1_f2_7 = Dense(2,activation='softmax',)(concat_7)
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

    concat_8 = concatenate([x1_8,x2_8] , axis=-1)

    output_f1_f2_8 = Dense(2,activation='softmax',)(concat_8)
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

    concat_9 = concatenate([x1_9,x2_9] , axis=-1)

    output_f1_f2_9 = Dense(2,activation='softmax',)(concat_9)
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

    concat_10 = concatenate([x1_10,x2_10] , axis=-1)

    output_f1_f2_10 = Dense(2,activation='softmax',)(concat_10)
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

    concat_11 = concatenate([x1_11,x2_11] , axis=-1)

    output_f1_f2_11 = Dense(2,activation='softmax',)(concat_11)
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

    concat_12 = concatenate([x1_12,x2_12] , axis=-1)

    output_f1_f2_12 = Dense(2,activation='softmax',)(concat_12)
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

    concat_13 = concatenate([x1_13,x2_13] , axis=-1)

    output_f1_f2_13 = Dense(2,activation='softmax',)(concat_13)
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

    concat_14 = concatenate([x1_14,x2_14] , axis=-1)

    output_f1_f2_14 = Dense(2,activation='softmax',)(concat_14)
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

    concat_15 = concatenate([x1_15,x2_15] , axis=-1)

    output_f1_f2_15 = Dense(2,activation='softmax',)(concat_15)
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

    concat_16 = concatenate([x1_16,x2_16] , axis=-1)

    output_f1_f2_16 = Dense(2,activation='softmax',)(concat_16)
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

    concat_17 = concatenate([x1_17,x2_17] , axis=-1)

    output_f1_f2_17 = Dense(2,activation='softmax',)(concat_17)
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

    concat_18 = concatenate([x1_18,x2_18] , axis=-1)

    output_f1_f2_18 = Dense(2,activation='softmax',)(concat_18)
#!################################################################################################################
    input_19_f1 = Input(shape=(1,), name='input_19_f1_layer')
    input_19_f2 = Input(shape=(1,), name='input_19_f2_layer')
    
    x1_19 = Embedding(input_dim=emb_mtx_f1_19.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_19],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_19_f1)
    x1_19 = Flatten()(x1_1)

    x2_19 = Embedding(input_dim=emb_mtx_f2_19.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_19],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_19_f2)
    x2_19 = Flatten()(x2_1)

    concat_19 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_19 = Dense(2,activation='softmax',)(concat_19)
#!################################################################################################################
    input_20_f1 = Input(shape=(1,), name='input_20_f1_layer')
    input_20_f2 = Input(shape=(1,), name='input_20_f2_layer')
    
    x1_20 = Embedding(input_dim=emb_mtx_f1_20.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_20],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_20_f1)
    x1_20 = Flatten()(x1_1)

    x2_20 = Embedding(input_dim=emb_mtx_f2_20.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_20],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_20_f2)
    x2_20 = Flatten()(x2_1)

    concat_20 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_20 = Dense(2,activation='softmax',)(concat_20)
#!################################################################################################################
    input_21_f1 = Input(shape=(1,), name='input_21_f1_layer')
    input_21_f2 = Input(shape=(1,), name='input_21_f2_layer')
    
    x1_21 = Embedding(input_dim=emb_mtx_f1_21.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_21],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_21_f1)
    x1_21 = Flatten()(x1_1)

    x2_21 = Embedding(input_dim=emb_mtx_f2_21.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_21],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_21_f2)
    x2_21 = Flatten()(x2_1)

    concat_21 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_21 = Dense(2,activation='softmax',)(concat_21)
#!################################################################################################################
    input_22_f1 = Input(shape=(1,), name='input_22_f1_layer')
    input_22_f2 = Input(shape=(1,), name='input_22_f2_layer')
    
    x1_22 = Embedding(input_dim=emb_mtx_f1_22.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_22],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_22_f1)
    x1_22 = Flatten()(x1_1)

    x2_22 = Embedding(input_dim=emb_mtx_f2_22.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_22],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_22_f2)
    x2_22 = Flatten()(x2_1)

    concat_22 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_22 = Dense(2,activation='softmax',)(concat_22)
#!################################################################################################################
    input_23_f1 = Input(shape=(1,), name='input_23_f1_layer')
    input_23_f2 = Input(shape=(1,), name='input_23_f2_layer')
    
    x1_23 = Embedding(input_dim=emb_mtx_f1_23.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_23],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_23_f1)
    x1_23 = Flatten()(x1_1)

    x2_23 = Embedding(input_dim=emb_mtx_f2_23.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_23],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_23_f2)
    x2_23 = Flatten()(x2_1)

    concat_23 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_23 = Dense(2,activation='softmax',)(concat_23)
#!################################################################################################################
    input_24_f1 = Input(shape=(1,), name='input_24_f1_layer')
    input_24_f2 = Input(shape=(1,), name='input_24_f2_layer')
    
    x1_24 = Embedding(input_dim=emb_mtx_f1_24.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_24],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_24_f1)
    x1_24 = Flatten()(x1_1)

    x2_24 = Embedding(input_dim=emb_mtx_f2_24.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_24],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_24_f2)
    x2_24 = Flatten()(x2_1)

    concat_24 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_24 = Dense(2,activation='softmax',)(concat_24)
#!################################################################################################################
    input_25_f1 = Input(shape=(1,), name='input_25_f1_layer')
    input_25_f2 = Input(shape=(1,), name='input_25_f2_layer')
    
    x1_25 = Embedding(input_dim=emb_mtx_f1_25.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_25],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_25_f1)
    x1_25 = Flatten()(x1_1)

    x2_25 = Embedding(input_dim=emb_mtx_f2_25.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_25],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_25_f2)
    x2_25 = Flatten()(x2_1)

    concat_25 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_25 = Dense(2,activation='softmax',)(concat_25)
#!################################################################################################################
    input_26_f1 = Input(shape=(1,), name='input_26_f1_layer')
    input_26_f2 = Input(shape=(1,), name='input_26_f2_layer')
    
    x1_26 = Embedding(input_dim=emb_mtx_f1_26.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_26],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_26_f1)
    x1_26 = Flatten()(x1_1)

    x2_26 = Embedding(input_dim=emb_mtx_f2_26.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_26],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_26_f2)
    x2_26 = Flatten()(x2_1)

    concat_26 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_26 = Dense(2,activation='softmax',)(concat_26)

#!################################################################################################################
    input_27_f1 = Input(shape=(1,), name='input_27_f1_layer')
    input_27_f2 = Input(shape=(1,), name='input_27_f2_layer')
    
    x1_27 = Embedding(input_dim=emb_mtx_f1_27.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_27],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_27_f1)
    x1_27 = Flatten()(x1_1)

    x2_27 = Embedding(input_dim=emb_mtx_f2_27.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_27],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_27_f2)
    x2_27 = Flatten()(x2_1)

    concat_27 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_27 = Dense(2,activation='softmax',)(concat_27)
#!################################################################################################################
    input_28_f1 = Input(shape=(1,), name='input_28_f1_layer')
    input_28_f2 = Input(shape=(1,), name='input_28_f2_layer')
    
    x1_28 = Embedding(input_dim=emb_mtx_f1_28.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_28],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_28_f1)
    x1_28 = Flatten()(x1_1)

    x2_28 = Embedding(input_dim=emb_mtx_f2_28.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_28],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_28_f2)
    x2_28 = Flatten()(x2_1)

    concat_28 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_28 = Dense(2,activation='softmax',)(concat_28)
#!################################################################################################################
    input_29_f1 = Input(shape=(1,), name='input_29_f1_layer')
    input_29_f2 = Input(shape=(1,), name='input_29_f2_layer')
    
    x1_29 = Embedding(input_dim=emb_mtx_f1_29.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_29],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_29_f1)
    x1_29 = Flatten()(x1_1)

    x2_29 = Embedding(input_dim=emb_mtx_f2_29.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_29],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_29_f2)
    x2_29 = Flatten()(x2_1)

    concat_29 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_29 = Dense(2,activation='softmax',)(concat_29)
#!################################################################################################################
    input_30_f1 = Input(shape=(1,), name='input_30_f1_layer')
    input_30_f2 = Input(shape=(1,), name='input_30_f2_layer')
    
    x1_30 = Embedding(input_dim=emb_mtx_f1_30.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_30],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_30_f1)
    x1_30 = Flatten()(x1_1)

    x2_30 = Embedding(input_dim=emb_mtx_f2_30.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_30],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_30_f2)
    x2_30 = Flatten()(x2_1)

    concat_30 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_30 = Dense(2,activation='softmax',)(concat_30)
#!################################################################################################################
    input_31_f1 = Input(shape=(1,), name='input_31_f1_layer')
    input_31_f2 = Input(shape=(1,), name='input_31_f2_layer')
    
    x1_31 = Embedding(input_dim=emb_mtx_f1_31.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_31],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_31_f1)
    x1_31 = Flatten()(x1_1)

    x2_31 = Embedding(input_dim=emb_mtx_f2_31.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_31],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_31_f2)
    x2_31 = Flatten()(x2_1)

    concat_31 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_31 = Dense(2,activation='softmax',)(concat_31)

#!################################################################################################################
    input_32_f1 = Input(shape=(1,), name='input_32_f1_layer')
    input_32_f2 = Input(shape=(1,), name='input_32_f2_layer')
    
    x1_32 = Embedding(input_dim=emb_mtx_f1_32.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_32],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_32_f1)
    x1_32 = Flatten()(x1_1)

    x2_32 = Embedding(input_dim=emb_mtx_f2_32.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_32],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_32_f2)
    x2_32 = Flatten()(x2_1)

    concat_32 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_32 = Dense(2,activation='softmax',)(concat_32)
#!################################################################################################################
    input_33_f1 = Input(shape=(1,), name='input_33_f1_layer')
    input_33_f2 = Input(shape=(1,), name='input_33_f2_layer')
    
    x1_33 = Embedding(input_dim=emb_mtx_f1_33.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_33],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_33_f1)
    x1_33 = Flatten()(x1_1)

    x2_33 = Embedding(input_dim=emb_mtx_f2_33.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_33],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_33_f2)
    x2_33 = Flatten()(x2_1)

    concat_33 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_33 = Dense(2,activation='softmax',)(concat_33)
#!################################################################################################################
    input_34_f1 = Input(shape=(1,), name='input_34_f1_layer')
    input_34_f2 = Input(shape=(1,), name='input_34_f2_layer')
    
    x1_34 = Embedding(input_dim=emb_mtx_f1_34.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_34],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_34_f1)
    x1_34 = Flatten()(x1_1)

    x2_34 = Embedding(input_dim=emb_mtx_f2_34.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_34],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_34_f2)
    x2_34 = Flatten()(x2_1)

    concat_34 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_34 = Dense(2,activation='softmax',)(concat_34)
#!################################################################################################################
    input_35_f1 = Input(shape=(1,), name='input_35_f1_layer')
    input_35_f2 = Input(shape=(1,), name='input_35_f2_layer')
    
    x1_35 = Embedding(input_dim=emb_mtx_f1_35.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f1_35],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_35_f1)
    x1_35 = Flatten()(x1_1)

    x2_35 = Embedding(input_dim=emb_mtx_f2_35.shape[0],
                   output_dim=64,
                   weights=[emb_mtx_f2_35],
                   trainable=False,
                   input_length=1,
                   mask_zero=True)(input_35_f2)
    x2_35 = Flatten()(x2_1)

    concat_35 = concatenate([x1_1,x2_1] , axis=-1)

    output_f1_f2_35 = Dense(2,activation='softmax',)(concat_35)
#!################################################################################################################

    f1_f2_output = concatenate([output_f1_f2_1,output_f1_f2_2] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_3] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_4] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_5] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_6] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_7] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_8] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_9] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_10] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_11] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_12] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_13] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_14] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_15] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_16] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_17] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_18] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_19] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_20] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_21] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_22] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_23] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_24] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_25] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_26] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_27] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_28] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_29] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_30] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_31] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_32] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_33] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_34] , axis=-1)
    f1_f2_output = concatenate([f1_f2_output,output_f1_f2_35] , axis=-1)

#!################################################################################################################

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)

    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    dnn_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)

    mix = concatenate([f1_f2_output,dnn_input] , axis=-1) #!#mix

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(mix)


    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

#!################################################################################################################
    
    model = Model(inputs=[input_1_f1,input_1_f2,
                        input_2_f1,input_2_f2,
                        input_3_f1,input_3_f2,
                        input_4_f1,input_4_f2,
                        input_5_f1,input_5_f2,
                        input_6_f1,input_6_f2,
                        input_7_f1,input_7_f2,
                        input_8_f1,input_8_f2,
                        input_9_f1,input_9_f2,
                        input_10_f1,input_10_f2,
                        input_11_f1,input_11_f2,
                        input_12_f1,input_12_f2,
                        input_13_f1,input_13_f2,
                        input_14_f1,input_14_f2,
                        input_15_f1,input_15_f2,
                        input_16_f1,input_16_f2,
                        input_17_f1,input_17_f2,
                        input_18_f1,input_18_f2,
                        input_19_f1,input_19_f2,
                        input_20_f1,input_20_f2,
                        input_21_f1,input_21_f2,
                        input_22_f1,input_22_f2,
                        input_23_f1,input_23_f2,
                        input_24_f1,input_24_f2,
                        input_25_f1,input_25_f2,
                        input_26_f1,input_26_f2,
                        input_27_f1,input_27_f2,
                        input_28_f1,input_28_f2,
                        input_29_f1,input_29_f2,
                        input_30_f1,input_30_f2,
                        input_31_f1,input_31_f2,
                        input_32_f1,input_32_f2,
                        input_33_f1,input_33_f2,
                        input_34_f1,input_34_f2,
                        input_35_f1,input_35_f2,
                        features], 
                outputs=[output])

    # model.compile(
    #     optimizer=optimizers.Adam(2.5e-4),
    #     loss={'prediction_layer': losses.binary_crossentropy},
    #     metrics=['AUC'])
    # return model
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(
        optimizer=optimizers.Adam(2.5e-4),
        loss={'prediction_layer': losses.binary_crossentropy},
        metrics=['AUC'])
    return parallel_model

if __name__ == '__main__':
#!################################################################################################################

    checkpoint = ModelCheckpoint("epoch_{epoch:02d}.hdf5", 
                                save_weights_only=True, 
                                monitor='val_loss', 
                                verbose=1,
                                save_best_only=False, 
                                mode='auto', period=1)
    earlystop_callback = EarlyStopping(monitor="val_AUC",
                                        min_delta=0.00001,
                                        patience=3,
                                        verbose=1,
                                        mode="max",
                                        baseline=None,
                                        restore_best_weights=True,)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_AUC',
                                                              factor=0.5,
                                                              patience=1,
                                                              min_lr=0.0000001)

#!################################################################################################################

    deepfm_data = Cache.reload_cache('CACHE_data_deepfm.pkl')
    label = Cache.reload_cache('CACHE_train_NONcmr.pkl')['label'].values
    
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    
    emb_mtx_f1_path = []
    for i in f1_f2_list:
        emb_mtx_f1_path.append(str(i[0])+'_'+str(i[1])+'_emb_mtx_f1.npy')
    emb_mtx_f1_1 = np.load(emb_mtx_f1_path[0],allow_pickle=True)
    emb_mtx_f1_2 = np.load(emb_mtx_f1_path[1],allow_pickle=True)
    emb_mtx_f1_3 = np.load(emb_mtx_f1_path[2],allow_pickle=True)
    emb_mtx_f1_4 = np.load(emb_mtx_f1_path[3],allow_pickle=True)
    emb_mtx_f1_5 = np.load(emb_mtx_f1_path[4],allow_pickle=True)
    emb_mtx_f1_6 = np.load(emb_mtx_f1_path[5],allow_pickle=True)
    emb_mtx_f1_7 = np.load(emb_mtx_f1_path[6],allow_pickle=True)
    emb_mtx_f1_8 = np.load(emb_mtx_f1_path[7],allow_pickle=True)
    emb_mtx_f1_9 = np.load(emb_mtx_f1_path[8],allow_pickle=True)
    emb_mtx_f1_10 = np.load(emb_mtx_f1_path[9],allow_pickle=True)
    emb_mtx_f1_11 = np.load(emb_mtx_f1_path[10],allow_pickle=True)
    emb_mtx_f1_12 = np.load(emb_mtx_f1_path[11],allow_pickle=True)
    emb_mtx_f1_13 = np.load(emb_mtx_f1_path[12],allow_pickle=True)
    emb_mtx_f1_14 = np.load(emb_mtx_f1_path[13],allow_pickle=True)
    emb_mtx_f1_15 = np.load(emb_mtx_f1_path[14],allow_pickle=True)
    emb_mtx_f1_16 = np.load(emb_mtx_f1_path[15],allow_pickle=True)
    emb_mtx_f1_17 = np.load(emb_mtx_f1_path[16],allow_pickle=True)
    emb_mtx_f1_18 = np.load(emb_mtx_f1_path[17],allow_pickle=True)
    emb_mtx_f1_19 = np.load(emb_mtx_f1_path[18],allow_pickle=True)
    emb_mtx_f1_20 = np.load(emb_mtx_f1_path[19],allow_pickle=True)
    emb_mtx_f1_21 = np.load(emb_mtx_f1_path[20],allow_pickle=True)
    emb_mtx_f1_22 = np.load(emb_mtx_f1_path[21],allow_pickle=True)
    emb_mtx_f1_23 = np.load(emb_mtx_f1_path[22],allow_pickle=True)
    emb_mtx_f1_24 = np.load(emb_mtx_f1_path[23],allow_pickle=True)
    emb_mtx_f1_25 = np.load(emb_mtx_f1_path[24],allow_pickle=True)
    emb_mtx_f1_26 = np.load(emb_mtx_f1_path[25],allow_pickle=True)
    emb_mtx_f1_27 = np.load(emb_mtx_f1_path[26],allow_pickle=True)
    emb_mtx_f1_28 = np.load(emb_mtx_f1_path[27],allow_pickle=True)
    emb_mtx_f1_29 = np.load(emb_mtx_f1_path[28],allow_pickle=True)
    emb_mtx_f1_30 = np.load(emb_mtx_f1_path[29],allow_pickle=True)
    emb_mtx_f1_31 = np.load(emb_mtx_f1_path[30],allow_pickle=True)
    emb_mtx_f1_32 = np.load(emb_mtx_f1_path[31],allow_pickle=True)
    emb_mtx_f1_33 = np.load(emb_mtx_f1_path[32],allow_pickle=True)
    emb_mtx_f1_34 = np.load(emb_mtx_f1_path[33],allow_pickle=True)
    emb_mtx_f1_35 = np.load(emb_mtx_f1_path[34],allow_pickle=True)

    emb_mtx_f2_path = []
    for i in f1_f2_list:
        emb_mtx_f2_path.append(str(i[0])+'_'+str(i[1])+'_embedding_matrix.npy')
    emb_mtx_f2_1 = np.load(emb_mtx_f2_path[0],allow_pickle=True)
    emb_mtx_f2_2 = np.load(emb_mtx_f2_path[1],allow_pickle=True)
    emb_mtx_f2_3 = np.load(emb_mtx_f2_path[2],allow_pickle=True)
    emb_mtx_f2_4 = np.load(emb_mtx_f2_path[3],allow_pickle=True)
    emb_mtx_f2_5 = np.load(emb_mtx_f2_path[4],allow_pickle=True)
    emb_mtx_f2_6 = np.load(emb_mtx_f2_path[5],allow_pickle=True)
    emb_mtx_f2_7 = np.load(emb_mtx_f2_path[6],allow_pickle=True)
    emb_mtx_f2_8 = np.load(emb_mtx_f2_path[7],allow_pickle=True)
    emb_mtx_f2_9 = np.load(emb_mtx_f2_path[8],allow_pickle=True)
    emb_mtx_f2_10 = np.load(emb_mtx_f2_path[9],allow_pickle=True)
    emb_mtx_f2_11 = np.load(emb_mtx_f2_path[10],allow_pickle=True)
    emb_mtx_f2_12 = np.load(emb_mtx_f2_path[11],allow_pickle=True)
    emb_mtx_f2_13 = np.load(emb_mtx_f2_path[12],allow_pickle=True)
    emb_mtx_f2_14 = np.load(emb_mtx_f2_path[13],allow_pickle=True)
    emb_mtx_f2_15 = np.load(emb_mtx_f2_path[14],allow_pickle=True)
    emb_mtx_f2_16 = np.load(emb_mtx_f2_path[15],allow_pickle=True)
    emb_mtx_f2_17 = np.load(emb_mtx_f2_path[16],allow_pickle=True)
    emb_mtx_f2_18 = np.load(emb_mtx_f2_path[17],allow_pickle=True)
    emb_mtx_f2_19 = np.load(emb_mtx_f2_path[18],allow_pickle=True)
    emb_mtx_f2_20 = np.load(emb_mtx_f2_path[19],allow_pickle=True)
    emb_mtx_f2_21 = np.load(emb_mtx_f2_path[20],allow_pickle=True)
    emb_mtx_f2_22 = np.load(emb_mtx_f2_path[21],allow_pickle=True)
    emb_mtx_f2_23 = np.load(emb_mtx_f2_path[22],allow_pickle=True)
    emb_mtx_f2_24 = np.load(emb_mtx_f2_path[23],allow_pickle=True)
    emb_mtx_f2_25 = np.load(emb_mtx_f2_path[24],allow_pickle=True)
    emb_mtx_f2_26 = np.load(emb_mtx_f2_path[25],allow_pickle=True)
    emb_mtx_f2_27 = np.load(emb_mtx_f2_path[26],allow_pickle=True)
    emb_mtx_f2_28 = np.load(emb_mtx_f2_path[27],allow_pickle=True)
    emb_mtx_f2_29 = np.load(emb_mtx_f2_path[28],allow_pickle=True)
    emb_mtx_f2_30 = np.load(emb_mtx_f2_path[29],allow_pickle=True)
    emb_mtx_f2_31 = np.load(emb_mtx_f2_path[30],allow_pickle=True)
    emb_mtx_f2_32 = np.load(emb_mtx_f2_path[31],allow_pickle=True)
    emb_mtx_f2_33 = np.load(emb_mtx_f2_path[32],allow_pickle=True)
    emb_mtx_f2_34 = np.load(emb_mtx_f2_path[33],allow_pickle=True)
    emb_mtx_f2_35 = np.load(emb_mtx_f2_path[34],allow_pickle=True)

    w2v_f1_train_path_list = []
    for i in f1_f2_list:
        w2v_f1_train_path_list.append(str(i[0])+'_'+str(i[1])+'_f1_train.npy')
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
        w2v_f2_train_path_list.append(str(i[0])+'_'+str(i[1])+'_f2_train.npy')
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

    # w2v_f1_test_path_list = []
    # for i in f1_f2_list:
    #     w2v_f1_test_path_list.append(str(i[0])+'_'+str(i[1])+'_f1_test.npy')
    # w2v_1_f1_test = np.load(w2v_f1_test_path_list[0])
    # w2v_2_f1_test = np.load(w2v_f1_test_path_list[1])
    # w2v_3_f1_test = np.load(w2v_f1_test_path_list[2])
    # w2v_4_f1_test = np.load(w2v_f1_test_path_list[3])
    # w2v_5_f1_test = np.load(w2v_f1_test_path_list[4])
    # w2v_6_f1_test = np.load(w2v_f1_test_path_list[5])
    # w2v_7_f1_test = np.load(w2v_f1_test_path_list[6])
    # w2v_8_f1_test = np.load(w2v_f1_test_path_list[7])
    # w2v_9_f1_test = np.load(w2v_f1_test_path_list[8])
    # w2v_10_f1_test = np.load(w2v_f1_test_path_list[9])
    # w2v_11_f1_test = np.load(w2v_f1_test_path_list[10])
    # w2v_12_f1_test = np.load(w2v_f1_test_path_list[11])
    # w2v_13_f1_test = np.load(w2v_f1_test_path_list[12])
    # w2v_14_f1_test = np.load(w2v_f1_test_path_list[13])
    # w2v_15_f1_test = np.load(w2v_f1_test_path_list[14])
    # w2v_16_f1_test = np.load(w2v_f1_test_path_list[15])
    # w2v_17_f1_test = np.load(w2v_f1_test_path_list[16])
    # w2v_18_f1_test = np.load(w2v_f1_test_path_list[17])
    # w2v_19_f1_test = np.load(w2v_f1_test_path_list[18])
    # w2v_20_f1_test = np.load(w2v_f1_test_path_list[19])
    # w2v_21_f1_test = np.load(w2v_f1_test_path_list[20])
    # w2v_22_f1_test = np.load(w2v_f1_test_path_list[21])
    # w2v_23_f1_test = np.load(w2v_f1_test_path_list[22])
    # w2v_24_f1_test = np.load(w2v_f1_test_path_list[23])
    # w2v_25_f1_test = np.load(w2v_f1_test_path_list[24])
    # w2v_26_f1_test = np.load(w2v_f1_test_path_list[25])
    # w2v_27_f1_test = np.load(w2v_f1_test_path_list[26])
    # w2v_28_f1_test = np.load(w2v_f1_test_path_list[27])
    # w2v_29_f1_test = np.load(w2v_f1_test_path_list[28])
    # w2v_30_f1_test = np.load(w2v_f1_test_path_list[29])
    # w2v_31_f1_test = np.load(w2v_f1_test_path_list[30])
    # w2v_32_f1_test = np.load(w2v_f1_test_path_list[31])
    # w2v_33_f1_test = np.load(w2v_f1_test_path_list[32])
    # w2v_34_f1_test = np.load(w2v_f1_test_path_list[33])
    # w2v_35_f1_test = np.load(w2v_f1_test_path_list[34])                

    # w2v_f2_test_path_list = []
    # for i in f1_f2_list:
    #     w2v_f2_test_path_list.append(str(i[0])+'_'+str(i[1])+'_f2_test.npy')
    # w2v_1_f2_test = np.load(w2v_f2_test_path_list[0])
    # w2v_2_f2_test = np.load(w2v_f2_test_path_list[1])
    # w2v_3_f2_test = np.load(w2v_f2_test_path_list[2])
    # w2v_4_f2_test = np.load(w2v_f2_test_path_list[3])
    # w2v_5_f2_test = np.load(w2v_f2_test_path_list[4])
    # w2v_6_f2_test = np.load(w2v_f2_test_path_list[5])
    # w2v_7_f2_test = np.load(w2v_f2_test_path_list[6])
    # w2v_8_f2_test = np.load(w2v_f2_test_path_list[7])
    # w2v_9_f2_test = np.load(w2v_f2_test_path_list[8])
    # w2v_10_f2_test = np.load(w2v_f2_test_path_list[9])
    # w2v_11_f2_test = np.load(w2v_f2_test_path_list[10])
    # w2v_12_f2_test = np.load(w2v_f2_test_path_list[11])
    # w2v_13_f2_test = np.load(w2v_f2_test_path_list[12])
    # w2v_14_f2_test = np.load(w2v_f2_test_path_list[13])
    # w2v_15_f2_test = np.load(w2v_f2_test_path_list[14])
    # w2v_16_f2_test = np.load(w2v_f2_test_path_list[15])
    # w2v_17_f2_test = np.load(w2v_f2_test_path_list[16])
    # w2v_18_f2_test = np.load(w2v_f2_test_path_list[17])
    # w2v_19_f2_test = np.load(w2v_f2_test_path_list[18])
    # w2v_20_f2_test = np.load(w2v_f2_test_path_list[19])
    # w2v_21_f2_test = np.load(w2v_f2_test_path_list[20])
    # w2v_22_f2_test = np.load(w2v_f2_test_path_list[21])
    # w2v_23_f2_test = np.load(w2v_f2_test_path_list[22])
    # w2v_24_f2_test = np.load(w2v_f2_test_path_list[23])
    # w2v_25_f2_test = np.load(w2v_f2_test_path_list[24])
    # w2v_26_f2_test = np.load(w2v_f2_test_path_list[25])
    # w2v_27_f2_test = np.load(w2v_f2_test_path_list[26])
    # w2v_28_f2_test = np.load(w2v_f2_test_path_list[27])
    # w2v_29_f2_test = np.load(w2v_f2_test_path_list[28])
    # w2v_30_f2_test = np.load(w2v_f2_test_path_list[29])
    # w2v_31_f2_test = np.load(w2v_f2_test_path_list[30])
    # w2v_32_f2_test = np.load(w2v_f2_test_path_list[31])
    # w2v_33_f2_test = np.load(w2v_f2_test_path_list[32])
    # w2v_34_f2_test = np.load(w2v_f2_test_path_list[33])
    # w2v_35_f2_test = np.load(w2v_f2_test_path_list[34])   


    
#!################################################################################################################

    sparse_features = ['uid','task_id','adv_id','creat_type_cd','adv_prim_id','dev_id',
                        'inter_type_cd','slot_id','spread_app_id','tags','app_first_class',
                        'app_second_class','city','device_name','device_size','career','gender',
                        'net_type','residence','his_app_size','emui_dev','his_on_shelf_time',
                        'list_time','up_membership_grade','consume_purchase','indu_name','pt_d',
                        'communication_onlinerate_1','communication_onlinerate_2','communication_onlinerate_3',
                        'communication_onlinerate_4','communication_onlinerate_5','communication_onlinerate_6',
                        'communication_onlinerate_7','communication_onlinerate_8','communication_onlinerate_9',
                        'communication_onlinerate_10','communication_onlinerate_11','communication_onlinerate_12',
                        'communication_onlinerate_13','communication_onlinerate_14','communication_onlinerate_15',
                        'communication_onlinerate_16','communication_onlinerate_17','communication_onlinerate_18',
                        'communication_onlinerate_19','communication_onlinerate_20','communication_onlinerate_21',
                        'communication_onlinerate_22','communication_onlinerate_23','communication_onlinerate_24']#e.g.:05db9164
    dense_features = ['age','city_rank','app_score','device_price','up_life_duration',
                        'membership_life_duration','communication_avgonline_30d']#e.g.:16836.0

    # deepfm_data[sparse_features] = deepfm_data[sparse_features].fillna('-1', )
    # deepfm_data[dense_features] = deepfm_data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        deepfm_data[feat] = lbe.fit_transform(deepfm_data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    deepfm_data[dense_features] = mms.fit_transform(deepfm_data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=deepfm_data[feat].nunique(),embedding_dim=4)
                           for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input deepfm_data for model
    # train, test = train_test_split(deepfm_data, test_size=0.2)
    deepfm_train = deepfm_data.head(41907133)
    deepfm_test = deepfm_data.tail(1000000)

    deepfm_train = {name:deepfm_train[name] for name in feature_names}
    deepfm_test = {name:deepfm_test[name] for name in feature_names}

    model = M(emb_mtx_f1_1,emb_mtx_f1_2,emb_mtx_f1_3,emb_mtx_f1_4,emb_mtx_f1_5,
                emb_mtx_f1_6,emb_mtx_f1_7,emb_mtx_f1_8,emb_mtx_f1_9,emb_mtx_f1_10,
                emb_mtx_f1_11,emb_mtx_f1_12,emb_mtx_f1_13,emb_mtx_f1_14,emb_mtx_f1_15,
                emb_mtx_f1_16,emb_mtx_f1_17,emb_mtx_f1_18,emb_mtx_f1_19,emb_mtx_f1_20,
                emb_mtx_f1_21,emb_mtx_f1_22,emb_mtx_f1_23,emb_mtx_f1_24,emb_mtx_f1_25,
                emb_mtx_f1_26,emb_mtx_f1_27,emb_mtx_f1_28,emb_mtx_f1_29,emb_mtx_f1_30,
                emb_mtx_f1_31,emb_mtx_f1_32,emb_mtx_f1_33,emb_mtx_f1_34,emb_mtx_f1_35,
                #!###
                emb_mtx_f2_1,emb_mtx_f2_2,emb_mtx_f2_3,emb_mtx_f2_4,emb_mtx_f2_5,
                emb_mtx_f2_6,emb_mtx_f2_7,emb_mtx_f2_8,emb_mtx_f2_9,emb_mtx_f2_10,
                emb_mtx_f2_11,emb_mtx_f2_12,emb_mtx_f2_13,emb_mtx_f2_14,emb_mtx_f2_15,
                emb_mtx_f2_16,emb_mtx_f2_17,emb_mtx_f2_18,emb_mtx_f2_19,emb_mtx_f2_20,
                emb_mtx_f2_21,emb_mtx_f2_22,emb_mtx_f2_23,emb_mtx_f2_24,emb_mtx_f2_25,
                emb_mtx_f2_26,emb_mtx_f2_27,emb_mtx_f2_28,emb_mtx_f2_29,emb_mtx_f2_30,
                emb_mtx_f2_31,emb_mtx_f2_32,emb_mtx_f2_33,emb_mtx_f2_34,emb_mtx_f2_35,
                #!###
                linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary')

    model.summary()

#!################################################################################################################

    input_train = {'input_1_f1_layer':w2v_1_f1_train,
                'input_1_f2_layer':w2v_1_f2_train,
                'input_2_f1_layer':w2v_2_f1_train,
                'input_2_f2_layer':w2v_2_f2_train,
                'input_3_f1_layer':w2v_3_f1_train,
                'input_3_f2_layer':w2v_3_f2_train,
                'input_4_f1_layer':w2v_4_f1_train,
                'input_4_f2_layer':w2v_4_f2_train,
                'input_5_f1_layer':w2v_5_f1_train,
                'input_5_f2_layer':w2v_5_f2_train,
                'input_6_f1_layer':w2v_6_f1_train,
                'input_6_f2_layer':w2v_6_f2_train,
                'input_7_f1_layer':w2v_7_f1_train,
                'input_7_f2_layer':w2v_7_f2_train,
                'input_8_f1_layer':w2v_8_f1_train,
                'input_8_f2_layer':w2v_8_f2_train,
                'input_9_f1_layer':w2v_9_f1_train,
                'input_9_f2_layer':w2v_9_f2_train,
                'input_10_f1_layer':w2v_10_f1_train,
                'input_10_f2_layer':w2v_10_f2_train,
                'input_11_f1_layer':w2v_11_f1_train,
                'input_11_f2_layer':w2v_11_f2_train,
                'input_12_f1_layer':w2v_12_f1_train,
                'input_12_f2_layer':w2v_12_f2_train,
                'input_13_f1_layer':w2v_13_f1_train,
                'input_13_f2_layer':w2v_13_f2_train,
                'input_14_f1_layer':w2v_14_f1_train,
                'input_14_f2_layer':w2v_14_f2_train,
                'input_15_f1_layer':w2v_15_f1_train,
                'input_15_f2_layer':w2v_15_f2_train,
                'input_16_f1_layer':w2v_16_f1_train,
                'input_16_f2_layer':w2v_16_f2_train,
                'input_17_f1_layer':w2v_17_f1_train,
                'input_17_f2_layer':w2v_17_f2_train,
                'input_18_f1_layer':w2v_18_f1_train,
                'input_18_f2_layer':w2v_18_f2_train,
                'input_19_f1_layer':w2v_19_f1_train,
                'input_19_f2_layer':w2v_19_f2_train,
                'input_20_f1_layer':w2v_20_f1_train,
                'input_20_f2_layer':w2v_20_f2_train,
                'input_21_f1_layer':w2v_21_f1_train,
                'input_21_f2_layer':w2v_21_f2_train,
                'input_22_f1_layer':w2v_22_f1_train,
                'input_22_f2_layer':w2v_22_f2_train,
                'input_23_f1_layer':w2v_23_f1_train,
                'input_23_f2_layer':w2v_23_f2_train,
                'input_24_f1_layer':w2v_24_f1_train,
                'input_24_f2_layer':w2v_24_f2_train,
                'input_25_f1_layer':w2v_25_f1_train,
                'input_25_f2_layer':w2v_25_f2_train,
                'input_26_f1_layer':w2v_26_f1_train,
                'input_26_f2_layer':w2v_26_f2_train,
                'input_27_f1_layer':w2v_27_f1_train,
                'input_27_f2_layer':w2v_27_f2_train,
                'input_28_f1_layer':w2v_28_f1_train,
                'input_28_f2_layer':w2v_28_f2_train,
                'input_29_f1_layer':w2v_29_f1_train,
                'input_29_f2_layer':w2v_29_f2_train,
                'input_30_f1_layer':w2v_30_f1_train,
                'input_30_f2_layer':w2v_30_f2_train,
                'input_31_f1_layer':w2v_31_f1_train,
                'input_31_f2_layer':w2v_31_f2_train,
                'input_32_f1_layer':w2v_32_f1_train,
                'input_32_f2_layer':w2v_32_f2_train,
                'input_33_f1_layer':w2v_33_f1_train,
                'input_33_f2_layer':w2v_33_f2_train,
                'input_34_f1_layer':w2v_34_f1_train,
                'input_34_f2_layer':w2v_34_f2_train,
                'input_35_f1_layer':w2v_35_f1_train,
                'input_35_f2_layer':w2v_35_f2_train}

    input_train.update(deepfm_train)
    model.fit(input_train,
            {'prediction_layer':label},
            validation_split=0.3,
            epochs=20,
            batch_size=5000,)
            # steps_per_epoch=1)
            # callbacks=[checkpoint, earlystop_callback, reduce_lr_callback])
    
    # input_test = {'input_1_f1_layer':w2v_1_f1_test,
    #             'input_1_f2_layer':w2v_1_f2_test,
    #             'input_2_f1_layer':w2v_2_f1_test,
    #             'input_2_f2_layer':w2v_2_f2_test,
    #             'input_3_f1_layer':w2v_3_f1_test,
    #             'input_3_f2_layer':w2v_3_f2_test,
    #             'input_4_f1_layer':w2v_4_f1_test,
    #             'input_4_f2_layer':w2v_4_f2_test,
    #             'input_5_f1_layer':w2v_5_f1_test,
    #             'input_5_f2_layer':w2v_5_f2_test,
    #             'input_6_f1_layer':w2v_6_f1_test,
    #             'input_6_f2_layer':w2v_6_f2_test,
    #             'input_7_f1_layer':w2v_7_f1_test,
    #             'input_7_f2_layer':w2v_7_f2_test,
    #             'input_8_f1_layer':w2v_8_f1_test,
    #             'input_8_f2_layer':w2v_8_f2_test,
    #             'input_9_f1_layer':w2v_9_f1_test,
    #             'input_9_f2_layer':w2v_9_f2_test,
    #             'input_10_f1_layer':w2v_10_f1_test,
    #             'input_10_f2_layer':w2v_10_f2_test,
    #             'input_11_f1_layer':w2v_11_f1_test,
    #             'input_11_f2_layer':w2v_11_f2_test,
    #             'input_12_f1_layer':w2v_12_f1_test,
    #             'input_12_f2_layer':w2v_12_f2_test,
    #             'input_13_f1_layer':w2v_13_f1_test,
    #             'input_13_f2_layer':w2v_13_f2_test,
    #             'input_14_f1_layer':w2v_14_f1_test,
    #             'input_14_f2_layer':w2v_14_f2_test,
    #             'input_15_f1_layer':w2v_15_f1_test,
    #             'input_15_f2_layer':w2v_15_f2_test,
    #             'input_16_f1_layer':w2v_16_f1_test,
    #             'input_16_f2_layer':w2v_16_f2_test,
    #             'input_17_f1_layer':w2v_17_f1_test,
    #             'input_17_f2_layer':w2v_17_f2_test,
    #             'input_18_f1_layer':w2v_18_f1_test,
    #             'input_18_f2_layer':w2v_18_f2_test,
    #             'input_19_f1_layer':w2v_19_f1_test,
    #             'input_19_f2_layer':w2v_19_f2_test,
    #             'input_20_f1_layer':w2v_20_f1_test,
    #             'input_20_f2_layer':w2v_20_f2_test,
    #             'input_21_f1_layer':w2v_21_f1_test,
    #             'input_21_f2_layer':w2v_21_f2_test,
    #             'input_22_f1_layer':w2v_22_f1_test,
    #             'input_22_f2_layer':w2v_22_f2_test,
    #             'input_23_f1_layer':w2v_23_f1_test,
    #             'input_23_f2_layer':w2v_23_f2_test,
    #             'input_24_f1_layer':w2v_24_f1_test,
    #             'input_24_f2_layer':w2v_24_f2_test,
    #             'input_25_f1_layer':w2v_25_f1_test,
    #             'input_25_f2_layer':w2v_25_f2_test,
    #             'input_26_f1_layer':w2v_26_f1_test,
    #             'input_26_f2_layer':w2v_26_f2_test,
    #             'input_27_f1_layer':w2v_27_f1_test,
    #             'input_27_f2_layer':w2v_27_f2_test,
    #             'input_28_f1_layer':w2v_28_f1_test,
    #             'input_28_f2_layer':w2v_28_f2_test,
    #             'input_29_f1_layer':w2v_29_f1_test,
    #             'input_29_f2_layer':w2v_29_f2_test,
    #             'input_30_f1_layer':w2v_30_f1_test,
    #             'input_30_f2_layer':w2v_30_f2_test,
    #             'input_31_f1_layer':w2v_31_f1_test,
    #             'input_31_f2_layer':w2v_31_f2_test,
    #             'input_32_f1_layer':w2v_32_f1_test,
    #             'input_32_f2_layer':w2v_32_f2_test,
    #             'input_33_f1_layer':w2v_33_f1_test,
    #             'input_33_f2_layer':w2v_33_f2_test,
    #             'input_34_f1_layer':w2v_34_f1_test,
    #             'input_34_f2_layer':w2v_34_f2_test,
    #             'input_35_f1_layer':w2v_35_f1_test,
    #             'input_35_f2_layer':w2v_35_f2_test}
    # input_test.update(deepfm_test)
    # ans_mtx = model.predict(input_test,
    #                         batch_size=4000)
    # np.save('ans_mtx.npy',ans_mtx)
    