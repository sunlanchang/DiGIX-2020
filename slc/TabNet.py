import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tabnet
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tabnet import TabNet, TabNetClassifier, StackedTabNetClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
train_size = 41907133
BATCH_SIZE = 1024
batch_size = 1024
epochs = 10
num_classes = 2
train_test = pd.read_pickle('data/train_test.pkl')
train = train_test.iloc[:train_size]
label = pd.read_pickle('data/label.pkl')
# train_label = pd.concat([train, label], ignore_index=True, axis=1)
# train_label = pd.concat([train, label], ignore_index=False, axis=1)
# x_train = train.iloc[:100, :]
# y_train = label.iloc[:100, :]
# x_val = train.iloc[100:200, :]
# y_val = label.iloc[100:200, :]


size = train.shape[0]//10
x_train = train.iloc[size:, :]
y_train = label.iloc[size:, :]
x_val = train.iloc[:size, :]
y_val = label.iloc[:size, :]

ds_full = tf.data.Dataset.from_tensor_slices({'features': x_train.astype(
    'u1').values, 'label': to_categorical(y_train).astype('u1')})
ds_full = ds_full.shuffle(2048, seed=0)

col_names = x_train.columns.tolist()


def transform(ds):
    features = tf.unstack(ds['features'])
    labels = ds['label']

    x = dict(zip(col_names, features))
#     y = tf.one_hot(labels, 2)
    y = labels
    return x, y


ds_train = ds_full.map(transform)
ds_train = ds_train.batch(BATCH_SIZE)

feature_columns = []
for col_name in col_names:
    feature_columns.append(tf.feature_column.numeric_column(col_name))

model = tabnet.StackedTabNetClassifier(feature_columns=feature_columns, num_classes=2,
                                       num_layers=5,
                                       num_features=len(feature_columns),
                                       norm_type='group',
                                       num_groups=1)
lr = tf.keras.optimizers.schedules.ExponentialDecay(
    0.01, decay_steps=100, decay_rate=0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['AUC'])

model.fit(ds_train, epochs=epochs, verbose=1)

# x_train = train.iloc[train.shape[0]//10:, :]
# y_train = label.iloc[train.shape[0]//10:, :]
# x_val = train.iloc[:train.shape[0]//10, :]
# y_val = label.iloc[:train.shape[0]//10, :]
# x_train, x_val, y_train, y_val = train_test_split(
#          train, label, test_size=0.1, random_state=42)
# x_train_list = x_train.to_dict(orient='list')

# x_val_list = x_val.to_dict(orient='list')
# %%
# ds_train = tf.data.Dataset.from_tensor_slices(
#     (x_train_list, to_categorical(y_train)))
# ds_val = tf.data.Dataset.from_tensor_slices(
#     (x_val_list, to_categorical(y_val)))
# ds_train = ds_train.shuffle(100).batch(batch_size)
# ds_test = ds_val.batch(batch_size)
# col_names = x_train.columns.tolist()
# feature_columns = []
# for col_name in col_names:
#     feature_columns.append(tf.feature_column.numeric_column(col_name))
# model = StackedTabNetClassifier(feature_columns=feature_columns, num_classes=num_classes,
#                                 num_layers=1,
#                                 num_features=len(feature_columns),
#                                 norm_type='group',
#                                 num_groups=1)
# lr = tf.keras.optimizers.schedules.ExponentialDecay(
#     0.001, decay_steps=500, decay_rate=0.9, staircase=False)
# optimizer = tf.keras.optimizers.Adam(lr)
# model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy','AUC'])

# model.fit(ds_train, epochs=epochs, validation_data=ds_train)
# model.summary()
