#%%
# 当前sota mask1版本的预训练行为蓄力（0.80448）-》加10组成对生成的emb（get_seq_emb）+slotidnettype 进dense-》0.80578
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
import pdb
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

# %%
data = Cache.reload_cache('CACHE_data_step_1_feature_0924_r5.pkl')
data.drop(columns=['communication_onlinerate'], inplace=True)

sparse_features = ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'device_name', 'career', 'gender', 'net_type', 'residence', 'emui_dev',
                   'indu_name', 'cmr_0', 'cmr_1', 'cmr_2', 'cmr_3', 'cmr_4', 'cmr_5', 'cmr_6', 'cmr_7', 'cmr_8', 'cmr_9', 'cmr_10', 'cmr_11', 'cmr_12', 'cmr_13', 'cmr_14', 'cmr_15', 'cmr_16', 'cmr_17', 'cmr_18', 'cmr_19', 'cmr_20', 'cmr_21', 'cmr_22', 'age', 'city_rank']
# 删除掉cmr_23
dense_features = [i for i in data.columns if i not in sparse_features +
                  ['index', 'id', 'uid', 'level_0', 'pt_d', 'label'] and i.find('_cemb') == -1]
dense_feature_size = len(dense_features)
# pdb.set_trace()
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

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=8)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                        for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print('feature_names finish!')


def get_input_feature_layer(name=None, feature_shape=dense_feature_size, dtype="float32"):
    input_layer = tf.keras.Input(
        shape=(feature_shape,), dtype=dtype, name=name)
    return input_layer


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
    # seq_inputs_dict = get_cross_seq_input_layers(cols=cross_arr_name_list)
    # inputs_all = inputs_all + list(seq_inputs_dict.values())  # 输入层list 做交叉

    # cross_emb_out = []
    # last_col = ''
    # for index, col in enumerate(cross_arr_name_list):
    #     #         print(col, 'get embedding!')
    #     emb_layer = get_emb_layer(
    #         col, trainable=False, emb_matrix=dict_cross_emb_all[col])
    #     x = emb_layer(inputs_all[1+index])
    #     if col.split('_')[-1] == 'i':
    #         cross_user_item_i = x
    #         last_col = col
    #         continue
    #     else:
    #         print(f'crossing net add {last_col} and {col}')
    #         cross_emb_out.append(
    #             cross_net(cross_user_item_i, x, layer_slotid_nettype, hidden_unit=4))
    # cross_emb_out = tf.keras.layers.concatenate(cross_emb_out)
    # cross_emb_out = tf.squeeze(cross_emb_out, [1])
#!################################################################################################################
    # seq_inputs_dict = get_seq_input_layers(cols=arr_name_list)
    # inputs_all = inputs_all+list(seq_inputs_dict.values())  # 输入层list
    # masks = tf.equal(seq_inputs_dict['task_id'], 0)
    # # 普通序列+label序列
    # layers2concat = []
    # for index, col in enumerate(arr_name_list):
    #     print(col, 'get embedding!')
    #     emb_layer = get_emb_layer(
    #         col, trainable=TRAINABLE_DICT[col], emb_matrix=id_list_dict_emb_all[col][1])
    #     x = emb_layer(seq_inputs_dict[col])
    #     if conv1d_info_dict[col] > -1:
    #         cov_layer = tf.keras.layers.Conv1D(filters=conv1d_info_dict[col],
    #                                            kernel_size=1,
    #                                            activation='relu')
    #         x = cov_layer(x)
    #     layers2concat.append(x)
    # x = tf.keras.layers.concatenate(layers2concat)
#!################################################################################################################
#!mix1
    # x = trans_net(x, masks, hidden_unit=256)
    # max_pool = tf.keras.layers.GlobalMaxPooling1D()
    # average_pool = tf.keras.layers.GlobalAveragePooling1D()
    # xmaxpool = max_pool(x)
    # xmeanpool = average_pool(x)

    # trans_output = tf.keras.layers.concatenate([xmaxpool, xmeanpool])


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

    # mix = concatenate([cross_emb_out, trans_output,
    #    dnn_input], axis=-1)  # !#mix
    mix = dnn_input

    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     dnn_use_bn, seed)(mix)

    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    final_logit = add_func([linear_logit, fm_logit, dnn_logit])
    output = PredictionLayer(task)(final_logit)

#!################################################################################################################

    # model = Model(inputs=inputs_all+[features],
    model = Model(inputs=inputs_list,
                  outputs=[output])
    print(model.summary())
    return model


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


bs = 2048+512
count = 0
random_state = 1111
set1 = data.query('pt_d<=7').copy()
set2 = data.query('pt_d==9').copy()
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
