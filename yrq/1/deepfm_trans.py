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

def M(emb1,emb1_label,emb2,emb2_label,emb3,emb3_label,emb4,emb4_label,emb5,emb5_label,
    linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
    l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
    dnn_activation='relu', dnn_use_bn=False, task='binary'):

#!################################################################################################################

    feed_forward_size_trans_1 = 2048
    max_seq_len_trans_1 = 40
    model_dim_trans_1 = 128

    input_trans_1 = Input(shape=(max_seq_len_trans_1,), name='input_trans_1_layer')
    input_trans_1_label = Input(shape=(max_seq_len_trans_1,), name='input_trans_1_label_layer')
    
    x = Embedding(input_dim=5307+1,
                   output_dim=128,
                   weights=[emb1],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_1)

    x_label = Embedding(input_dim=2+1,
                   output_dim=128,
                   weights=[emb1_label],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_1_label)

    encodings = PositionEncoding(model_dim_trans_1)(x)
    encodings = Add()([x, encodings])
    encodings = Add()([x_label, encodings])

    # encodings = x
    masks = tf.equal(input_trans_1, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(4, 32)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim_trans_1, feed_forward_size_trans_1)
    ff_out = ff(attention_out)
    # Add & Norm
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_trans_1 = Dense(5, activation='softmax', name='output_trans_1_layer')(encodings)

#!################################################################################################################

    feed_forward_size_trans_2 = 2048
    max_seq_len_trans_2 = 40
    model_dim_trans_2 = 128

    input_trans_2 = Input(shape=(max_seq_len_trans_2,), name='input_trans_2_layer')
    input_trans_2_label = Input(shape=(max_seq_len_trans_2,), name='input_trans_2_label_layer')
    
    x = Embedding(input_dim=101+1,
                   output_dim=128,
                   weights=[emb2],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_2)

    x_label = Embedding(input_dim=2+1,
                   output_dim=128,
                   weights=[emb2_label],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_2_label)

    encodings = PositionEncoding(model_dim_trans_2)(x)
    encodings = Add()([x, encodings])
    encodings = Add()([x_label, encodings])

    # encodings = x
    masks = tf.equal(input_trans_2, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(4, 32)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim_trans_2, feed_forward_size_trans_2)
    ff_out = ff(attention_out)
    # Add & Norm
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_trans_2 = Dense(5, activation='softmax', name='output_trans_2_layer')(encodings)
    
#!################################################################################################################

    feed_forward_size_trans_3 = 2048
    max_seq_len_trans_3 = 40
    model_dim_trans_3 = 128

    input_trans_3 = Input(shape=(max_seq_len_trans_3,), name='input_trans_3_layer')
    input_trans_3_label = Input(shape=(max_seq_len_trans_3,), name='input_trans_3_label_layer')
    
    x = Embedding(input_dim=8+1,
                   output_dim=128,
                   weights=[emb3],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_3)

    x_label = Embedding(input_dim=2+1,
                   output_dim=128,
                   weights=[emb3_label],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_3_label)

    encodings = PositionEncoding(model_dim_trans_3)(x)
    encodings = Add()([x, encodings])
    encodings = Add()([x_label, encodings])

    # encodings = x
    masks = tf.equal(input_trans_3, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(4, 32)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim_trans_3, feed_forward_size_trans_3)
    ff_out = ff(attention_out)
    # Add & Norm
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_trans_3 = Dense(5, activation='softmax', name='output_trans_3_layer')(encodings)
    
#!################################################################################################################

    feed_forward_size_trans_4 = 2048
    max_seq_len_trans_4 = 40
    model_dim_trans_4 = 128

    input_trans_4 = Input(shape=(max_seq_len_trans_4,), name='input_trans_4_layer')
    input_trans_4_label = Input(shape=(max_seq_len_trans_4,), name='input_trans_4_label_layer')
    
    x = Embedding(input_dim=38+1,
                   output_dim=128,
                   weights=[emb4],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_4)

    x_label = Embedding(input_dim=2+1,
                   output_dim=128,
                   weights=[emb4_label],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_4_label)

    encodings = PositionEncoding(model_dim_trans_4)(x)
    encodings = Add()([x, encodings])
    encodings = Add()([x_label, encodings])

    # encodings = x
    masks = tf.equal(input_trans_4, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(4, 32)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim_trans_4, feed_forward_size_trans_4)
    ff_out = ff(attention_out)
    # Add & Norm
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_trans_4 = Dense(5, activation='softmax', name='output_trans_4_layer')(encodings)
    
#!################################################################################################################

    feed_forward_size_trans_5 = 2048
    max_seq_len_trans_5 = 40
    model_dim_trans_5 = 128

    input_trans_5 = Input(shape=(max_seq_len_trans_5,), name='input_trans_5_layer')
    input_trans_5_label = Input(shape=(max_seq_len_trans_5,), name='input_trans_5_label_layer')
    
    x = Embedding(input_dim=4317+1,
                   output_dim=128,
                   weights=[emb5],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_5)

    x_label = Embedding(input_dim=2+1,
                   output_dim=128,
                   weights=[emb5_label],
                   trainable=False,
                   input_length=40,
                   mask_zero=True)(input_trans_5_label)

    encodings = PositionEncoding(model_dim_trans_5)(x)
    encodings = Add()([x, encodings])
    encodings = Add()([x_label, encodings])

    # encodings = x
    masks = tf.equal(input_trans_5, 0)

    # (bs, 100, 128*2)
    attention_out = MultiHeadAttention(4, 32)(
        [encodings, encodings, encodings, masks])

    # Add & Norm
    attention_out += encodings
    attention_out = LayerNormalization()(attention_out)
    # Feed-Forward
    ff = PositionWiseFeedForward(model_dim_trans_5, feed_forward_size_trans_5)
    ff_out = ff(attention_out)
    # Add & Norm
    ff_out += attention_out
    encodings = LayerNormalization()(ff_out)
    encodings = GlobalMaxPooling1D()(encodings)
    encodings = Dropout(0.2)(encodings)

    output_trans_5 = Dense(5, activation='softmax', name='output_trans_5_layer')(encodings)
    
#!################################################################################################################

    trans_output = concatenate([output_trans_1,output_trans_2] , axis=-1)
    trans_output = concatenate([trans_output,output_trans_3] , axis=-1)
    trans_output = concatenate([trans_output,output_trans_4] , axis=-1)
    trans_output = concatenate([trans_output,output_trans_5] , axis=-1)
    # trans_output = Dense(2, activation='softmax', name='output_trans')(trans_output)

#!################################################################################################################
#!mix2
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

    mix = concatenate([trans_output,dnn_input] , axis=-1) #!#mix

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(mix)


    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

#!################################################################################################################
    
    model = Model(inputs=[input_trans_1,
                        input_trans_1_label,
                        input_trans_2,
                        input_trans_2_label,
                        input_trans_3,
                        input_trans_3_label,
                        input_trans_4,
                        input_trans_4_label,
                        input_trans_5,
                        input_trans_5_label,
                        features], 
                outputs=[output])

    model.compile(
        optimizer=optimizers.Adam(2.5e-4),
        loss={'prediction_layer': losses.binary_crossentropy},
        metrics=['AUC'])
    return model

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
                                                           
    emb1 = np.load('adv_idembedding_matrix.npy', allow_pickle=True)
    emb2 = np.load('adv_prim_idembedding_matrix.npy', allow_pickle=True)
    emb3 = np.load('creat_type_cdembedding_matrix.npy', allow_pickle=True)
    emb4 = np.load('indu_nameembedding_matrix.npy', allow_pickle=True)
    emb5 = np.load('task_idembedding_matrix.npy', allow_pickle=True)
    emb_label = np.load('labelembedding_matrix.npy', allow_pickle=True)

    trans_1_train = np.load('adv_idx_train.npy', allow_pickle=True)
    trans_2_train = np.load('adv_prim_idx_train.npy', allow_pickle=True)
    trans_3_train = np.load('creat_type_cdx_train.npy', allow_pickle=True)
    trans_4_train = np.load('indu_namex_train.npy', allow_pickle=True)
    trans_5_train = np.load('task_idx_train.npy', allow_pickle=True)
    trans_label_train = np.load('labelx_train.npy', allow_pickle=True)

    trans_1_test = np.load('adv_idx_test.npy', allow_pickle=True)
    trans_2_test = np.load('adv_prim_idx_test.npy', allow_pickle=True)
    trans_3_test = np.load('creat_type_cdx_test.npy', allow_pickle=True)
    trans_4_test = np.load('indu_namex_test.npy', allow_pickle=True)
    trans_5_test = np.load('task_idx_test.npy', allow_pickle=True)
    trans_label_test = np.load('labelx_test.npy', allow_pickle=True)

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

    model = M(emb1,emb_label,emb2,emb_label,emb3,emb_label,emb4,emb_label,emb5,emb_label,
            linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary')
    model.summary()

#!################################################################################################################

    input_train = {'input_trans_1_layer': trans_1_train,
            'input_trans_1_label_layer': trans_label_train,
            'input_trans_2_layer': trans_2_train,
            'input_trans_2_label_layer': trans_label_train,
            'input_trans_3_layer': trans_3_train,
            'input_trans_3_label_layer': trans_label_train,
            'input_trans_4_layer': trans_4_train,
            'input_trans_4_label_layer': trans_label_train,
            'input_trans_5_layer': trans_5_train,
            'input_trans_5_label_layer': trans_label_train}
    input_train.update(deepfm_train)
    model.fit(input_train,
            {'prediction_layer':label},
            # validation_split=0.3,
            epochs=5,
            batch_size=1000,)
            # steps_per_epoch=1)
            # callbacks=[checkpoint, earlystop_callback, reduce_lr_callback])
    
    input_test = {'input_trans_1_layer': trans_1_test,
            'input_trans_1_label_layer': trans_label_test,
            'input_trans_2_layer': trans_2_test,
            'input_trans_2_label_layer': trans_label_test,
            'input_trans_3_layer': trans_3_test,
            'input_trans_3_label_layer': trans_label_test,
            'input_trans_4_layer': trans_4_test,
            'input_trans_4_label_layer': trans_label_test,
            'input_trans_5_layer': trans_5_test,
            'input_trans_5_label_layer': trans_label_test}
    input_test.update(deepfm_test)
    ans_mtx = model.predict(input_test,
                            batch_size=4000)
    np.save('ans_mtx.npy',ans_mtx)
    