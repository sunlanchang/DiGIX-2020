<<<<<<< d06705dab73bdd5d47ba389418365e9c9dc926b6
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# from layers import Add, LayerNormalization
# from layers import MultiHeadAttention, PositionWiseFeedForward
# from layers import PositionEncoding

import tensorflow as tf
import keras.backend as K
import keras
# from tensorflow import keras
from keras import layers
from keras import losses
from keras import optimizers
from keras.callbacks import Callback
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, LSTM, Embedding, Dense, Dropout, Concatenate, Bidirectional, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import multi_gpu_model


from gensim.models import Word2Vec, KeyedVectors
import argparse

NUM_WORKERS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # allocate dynamically

parser = argparse.ArgumentParser()
parser.add_argument('--num_transformer', type=int,
                    help='transformer层数',
                    default=1)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='从npy文件加载数据',
                    default=True)
args = parser.parse_args()


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


def Adding_Layer(tensor):
    x, y = tensor
    return x + y


def trans_net(input, n_unit=512):
    input = keras.layers.Dropout(0.3)(input)
    encodings = keras.layers.Conv1D(
        filters=input.shape[-1].value, kernel_size=1, padding='same', activation='relu')(input)

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


def create_model():

    emb = np.load('word2vec/embedding_matrix.npy', allow_pickle=True)

    feed_forward_size = 2048
    max_seq_len = 36
    model_dim = 512

    input_x = Input(shape=(max_seq_len,), name='input_layer')

    x = Embedding(input_dim=1187210+1,
                  output_dim=512,
                  weights=[emb],
                  trainable=False,
                  input_length=max_seq_len,
                  mask_zero=False)(input_x)

    x8 = keras.layers.Conv1D(filters=256, kernel_size=1,
                             padding='same', activation='relu')(x)
    x = trans_net(x, n_unit=512)

    x = keras.layers.concatenate([x, x8])
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv1D(
        filters=x.shape[-1].value, kernel_size=1, padding='same', activation='relu')(x)
    lstm = keras.layers.Bidirectional(
        keras.layers.CuDNNLSTM(128, return_sequences=True))(x)
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

    outputs_y = keras.layers.Dense(
        2, activation='softmax', name='output_layer')(x)
    model = keras.Model(input_x, outputs_y)
    model.summary()

    return model


# def get_model2(emb):

#     feed_forward_size = 2048
#     max_seq_len = 36
#     model_dim = 512

#     input_x = Input(shape=(max_seq_len,), name='input_layer')

#     x = Embedding(input_dim=1187210+1,
#                    output_dim=512,
#                    weights=[emb],
#                    trainable=False,
#                    input_length=max_seq_len,
#                    mask_zero=True)(input_x)

# #     encodings = PositionEncoding(model_dim)(x)
# #     encodings = Add()([x, encodings])
#     import ipdb
#     ipdb.set_trace()
#     encodings = x
#     masks = tf.equal(input_x, 0)

#     # (bs, 100, 128*2)
#     attention_out = MultiHeadAttention(8, 64)(
#         [encodings, encodings, encodings, masks])

#     # Add & Norm
#     attention_out += encodings
#     attention_out = LayerNormalization()(attention_out)
#     # Feed-Forward
#     ff = PositionWiseFeedForward(model_dim, feed_forward_size)
#     ff_out = ff(attention_out)
#     # Add & Norm
#     ff_out += attention_out
#     encodings = LayerNormalization()(ff_out)
#     encodings = GlobalMaxPooling1D()(encodings)
#     encodings = Dropout(0.2)(encodings)

#     output_y = Dense(2, activation='softmax', name='output_layer')(encodings)

#     model = Model(inputs=[input_x], outputs=[output_y])


#     return model


def get_callbacks():

    checkpoint = keras.callbacks.ModelCheckpoint("epoch_{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=False,
                                                 mode='auto',
                                                 period=1)

    earlystop = keras.callbacks.EarlyStopping(
        monitor="val_AUC",
        min_delta=0.00001,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        #     restore_best_weights=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_AUC',
                                                  factor=0.5,
                                                  patience=5,
                                                  min_lr=0.0000001)
    return [checkpoint, earlystop, reduce_lr]


x_train = np.load('data/x_train.npy', allow_pickle=True)
label = np.load('data/label.npy', allow_pickle=True)

model = create_model()
model = multi_gpu_model(model, NUM_WORKERS)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model.compile(
    optimizer=optimizers.Adam(3e-4),
    loss='categorical_crossentropy',
    metrics=['acc', auc, f1])


BATCH_SIZE = 4096 * NUM_WORKERS
model.fit(
    x_train,
    label,
    validation_split=0.1,
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks()
)
=======
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

# from layers import Add, LayerNormalization
# from layers import MultiHeadAttention, PositionWiseFeedForward
# from layers import PositionEncoding

import tensorflow as tf
import keras.backend as K
import keras
# from tensorflow import keras
from keras import layers
from keras import losses
from keras import optimizers
from keras.callbacks import Callback
import keras.backend as K
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, LSTM, Embedding, Dense, Dropout, Concatenate, Bidirectional, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import multi_gpu_model


from gensim.models import Word2Vec, KeyedVectors
import argparse

NUM_WORKERS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # allocate dynamically

parser = argparse.ArgumentParser()
parser.add_argument('--num_transformer', type=int,
                    help='transformer层数',
                    default=1)
parser.add_argument('--not_train_embedding', action='store_false',
                    help='从npy文件加载数据',
                    default=True)
args = parser.parse_args()


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


def Adding_Layer(tensor):
    x, y = tensor
    return x + y


def trans_net(input, n_unit=512):
    input = keras.layers.Dropout(0.3)(input)
    encodings = keras.layers.Conv1D(
        filters=input.shape[-1].value, kernel_size=1, padding='same', activation='relu')(input)

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


def create_model():

    emb = np.load('word2vec/embedding_matrix.npy', allow_pickle=True)

    feed_forward_size = 2048
    max_seq_len = 36
    model_dim = 512

    input_x = Input(shape=(max_seq_len,), name='input_layer')

    x = Embedding(input_dim=1187210+1,
                  output_dim=512,
                  weights=[emb],
                  trainable=False,
                  input_length=max_seq_len,
                  mask_zero=False)(input_x)

    x8 = keras.layers.Conv1D(filters=256, kernel_size=1,
                             padding='same', activation='relu')(x)
    x = trans_net(x, n_unit=512)

    x = keras.layers.concatenate([x, x8])
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv1D(
        filters=x.shape[-1].value, kernel_size=1, padding='same', activation='relu')(x)
    lstm = keras.layers.Bidirectional(
        keras.layers.CuDNNLSTM(128, return_sequences=True))(x)
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

    outputs_y = keras.layers.Dense(
        2, activation='softmax', name='output_layer')(x)
    model = keras.Model(input_x, outputs_y)
    model.summary()

    return model


# def get_model2(emb):

#     feed_forward_size = 2048
#     max_seq_len = 36
#     model_dim = 512

#     input_x = Input(shape=(max_seq_len,), name='input_layer')

#     x = Embedding(input_dim=1187210+1,
#                    output_dim=512,
#                    weights=[emb],
#                    trainable=False,
#                    input_length=max_seq_len,
#                    mask_zero=True)(input_x)

# #     encodings = PositionEncoding(model_dim)(x)
# #     encodings = Add()([x, encodings])
#     import ipdb
#     ipdb.set_trace()
#     encodings = x
#     masks = tf.equal(input_x, 0)

#     # (bs, 100, 128*2)
#     attention_out = MultiHeadAttention(8, 64)(
#         [encodings, encodings, encodings, masks])

#     # Add & Norm
#     attention_out += encodings
#     attention_out = LayerNormalization()(attention_out)
#     # Feed-Forward
#     ff = PositionWiseFeedForward(model_dim, feed_forward_size)
#     ff_out = ff(attention_out)
#     # Add & Norm
#     ff_out += attention_out
#     encodings = LayerNormalization()(ff_out)
#     encodings = GlobalMaxPooling1D()(encodings)
#     encodings = Dropout(0.2)(encodings)

#     output_y = Dense(2, activation='softmax', name='output_layer')(encodings)

#     model = Model(inputs=[input_x], outputs=[output_y])


#     return model


def get_callbacks():

    checkpoint = keras.callbacks.ModelCheckpoint("epoch_{epoch:02d}.hdf5",
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=False,
                                                 mode='auto',
                                                 period=1)

    earlystop = keras.callbacks.EarlyStopping(
        monitor="val_AUC",
        min_delta=0.00001,
        patience=3,
        verbose=1,
        mode="max",
        baseline=None,
        #     restore_best_weights=True,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_AUC',
                                                  factor=0.5,
                                                  patience=5,
                                                  min_lr=0.0000001)
    return [checkpoint, earlystop, reduce_lr]


x_train = np.load('data/x_train.npy', allow_pickle=True)
label = np.load('data/label.npy', allow_pickle=True)

model = create_model()
model = multi_gpu_model(model, NUM_WORKERS)


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model.compile(
    optimizer=optimizers.Adam(3e-4),
    loss='categorical_crossentropy',
    metrics=['acc', auc, f1])


BATCH_SIZE = 4096 * NUM_WORKERS
model.fit(
    x_train,
    label,
    validation_split=0.1,
    epochs=20,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks()
)
>>>>>>> zlh first commit
