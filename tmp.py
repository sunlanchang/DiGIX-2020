import shutil
from pathlib import Path
import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.utils import to_categorical
import keras.backend as K
from tensorflow.keras.utils import multi_gpu_model

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from deepctr.models import DeepFM, FiBiNET
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
# from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names

import gc
import pickle
import time
import argparse
from tqdm import tqdm
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
os.environ["TF_KERAS"] = '1'

SEED = 123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # allocate dynamically

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    help='从npy文件加载数据',
                    default=False)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='从npy文件加载数据',
                    default=True)
parser.add_argument('--batch_size', type=int,
                    help='batch size大小',
                    default=256)
parser.add_argument('--epoch', type=int,
                    help='epoch 大小',
                    default=5)
parser.add_argument('--num_transformer', type=int,
                    help='transformer层数',
                    default=1)
parser.add_argument('--head_attention', type=int,
                    help='transformer head个数',
                    default=1)
parser.add_argument('--num_lstm', type=int,
                    help='LSTM 个数',
                    default=1)
parser.add_argument('--train_examples', type=int,
                    help='训练数据，默认为训练集，不包含验证集，调试时候可以设置1000',
                    default=810000)
parser.add_argument('--val_examples', type=int,
                    help='验证集数据，调试时候可以设置1000',
                    default=90000)
args = parser.parse_args()

LR = []


def get_callbacks():

    checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoint/epoch_{epoch:02d}.hdf5",
                                                    save_weights_only=True,
                                                    monitor='val_auroc',
                                                    verbose=1,
                                                    save_best_only=False,
                                                    mode='max',
                                                    period=1)

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auroc",
        patience=9,
        verbose=1,
        mode="max",
        # restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auroc',
                                                     factor=0.9,
                                                     verbose=1,
                                                     patience=1,
                                                     mode='max')

    def scheduler(epoch, lr):
        global LR
        LR.append(lr)
        return lr

    lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return [earlystop, reduce_lr, checkpoint, lr]


def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(
         alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
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
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                    100 *
                                                                                                    (start_mem-end_mem)/start_mem,
                                                                                                    (time.time()-starttime)/60))
    return df


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


train_df = pd.read_csv('data/train_data.csv', sep='|', nrows=500000)
# train_df = train_df.sample(frac=0.5).reset_index(drop=True)
test_df = pd.read_csv('data/test_data_A.csv', sep='|', nrows=500)
df = pd.concat([train_df, test_df], axis=0)
test_id = test_df['id'].copy().reset_index(drop=True)

df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

cate_cols = ['city_rank', 'creat_type_cd', 'dev_id', 'device_size', 'gender', 'indu_name', 'inter_type_cd', 'residence', 'slot_id', 'net_type',
             'task_id', 'adv_id', 'adv_prim_id', 'age', 'app_first_class', 'app_second_class', 'career', 'city', 'consume_purchase', 'uid', 'dev_id', 'tags']

for f in tqdm(cate_cols):
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = reduce_mem(df)


drop_fea = ['pt_d', 'label', 'communication_onlinerate',
            'index', 'uid', 'dev_id']
feature = [x for x in df.columns if x not in drop_fea]
print(len(feature))
print(feature)

sparse_features = cate_cols
# 这里的dense_feature可以把树模型的特征加进来
dense_features = [x for x in df.columns if x not in drop_fea+cate_cols]
print('sparse_feature: {}'.format(sparse_features))
print('dense_feature: {}'.format(dense_features))


# 对dense_features进行归一化
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=8)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                        for feat in dense_features]

val_data = df[df["pt_d"] == 7]
train_data = df[df["pt_d"] <= 7]
test_data = df[df['pt_d'] == 8]

# test_df = df[df["pt_d"]==8].copy().reset_index()
# train_df = df[df["pt_d"]<8].reset_index()
# del df
# gc.collect()

y_val = train_df['label'].iloc[val_data.index.tolist()]
y_train = train_df['label'].iloc[train_data.index.tolist()]


dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
online_train_model_input = {
    name: train_data[name].values for name in feature_names}
online_val_model_input = {
    name: val_data[name].values for name in feature_names}
online_test_model_input = {
    name: test_data[name].values for name in feature_names}


NUM_WORKERS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
BATCH_SIZE = 4096
EPOCHS = 2

N_FOLDS = 5
CV_SEED = 0

# train_and_val_model_input = {name: x_train_val[name] for name in feature_names}
# with tf.device("/cpu:0"):
model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary',
                dnn_dropout=0.1,
                dnn_hidden_units=(512, 128),)
# model = multi_gpu_model(model, NUM_WORKERS)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    # loss="binary_crossentropy",
    loss=multi_category_focal_loss2(alpha=0.1),
    metrics=[auroc], )

dirpath = Path('checkpoint')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
os.mkdir('checkpoint')


hist = model.fit(online_train_model_input, y_train.values,
                 validation_data=(online_val_model_input,
                                  y_val.values),
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 verbose=2,
                 #  validation_split=0.1,
                 shuffle=True,
                 callbacks=get_callbacks()
                 )
print('ssssss'*10)
print(LR)

best_epoch = np.argmax(hist.history["val_auroc"])+1
model.load_weights('checkpoint/epoch_{:02d}.hdf5'.format(best_epoch))
print(hist.history["val_auroc"])
print('loading epoch_{:02d}.hdf5'.format(best_epoch))

y_pre = model.predict(
    online_test_model_input, verbose=1, batch_size=BATCH_SIZE)
res = pd.DataFrame()
res['id'] = test_id
res['probability'] = y_pre
res.to_csv('submission_test1.csv', index=False)


# pred_ans = pred_ans.flatten()
# ans = pd.DataFrame(data={'id': np.array(
#     [i for i in range(1, pred_ans.shape[0]+1)]), 'probability': pred_ans})
# ans.to_csv('submission_DeepFM.csv', index=False, header=True)

# del model
# gc.collect()


# # EPOCHS = np.argmax(hist.history["val_auroc"])+1


# # # K折训练模型


# skf = StratifiedKFold(n_splits=N_FOLDS, random_state=CV_SEED, shuffle=True)
# folds = list(skf.split(x_train_val, y_train_val))


# score_val = []
# model_prob = np.zeros((x_test.shape[0], 1, N_FOLDS), dtype='float32')

# for count, (train_index, val_index) in enumerate(folds):

#     print(f'{count} fold start...')
#     print('#'*100)

#     x_train = x_train_val.iloc[train_index]
#     y_train = y_train_val.iloc[train_index]
#     x_val = x_train_val.iloc[val_index]
#     y_val = y_train_val.iloc[val_index]

#     val_model_input = {name: x_val[name] for name in feature_names}

#     model = DeepFM(linear_feature_columns=linear_feature_columns,
#                    dnn_feature_columns=dnn_feature_columns,
#                    dnn_dropout=0.1,
#                    dnn_hidden_units=(512, 128),
#                    task='binary')
#     model.compile(
#         optimizer=keras.optimizers.Adam(1e-3),
#         loss="binary_crossentropy",
#         #         loss=multi_category_focal_loss2(alpha=0.1),
#         metrics=[auroc], )

#     hist = model.fit(train_model_input, y_train,
#                      validation_data=(val_model_input, y_val),
#                      batch_size=BATCH_SIZE,
#                      epochs=EPOCHS,
#                      verbose=1,
#                      #                 validation_split=0.1,
#                      callbacks=get_callbacks()+[]
#                      )

#     pred_ans = model.predict(
#         test_model_input, verbose=1, batch_size=BATCH_SIZE)
#     model_prob[:, :, count] = pred_ans
#     del train_model_input, val_model_input
#     gc.collect()
#     try:
#         del model
#         gc.collect()
#         K.clear_session()
#     except Exception as e:
#         print(e)
#     print("v" * 100)
#     print(hist.history)
#     print("^" * 100)
#     score_val.append(np.max(hist.history["val_auroc"]))

# print(f"offline score: {score_val}")
# print(f"offline score by folds: {np.mean(score_val)}")
# print("All Done!")


# model_prob = np.mean(model_prob, axis=2).flatten()
# ans = pd.DataFrame(data={'id': np.array(
#     [i for i in range(1, 1000000+1)]), 'probability': model_prob})
# ans.to_csv('submission_DeepFM_k5.csv', index=False, header=True)
