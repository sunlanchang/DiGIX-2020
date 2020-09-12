{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:root:\n",
      "DeepCTR version 0.7.5 detected. Your version is 0.7.4.\n",
      "Use `pip install -U deepctr` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR/releases/tag/v0.7.5\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "features finish\n",
      "process 1 finish\n",
      "max_len: 200\n",
      "modeling!\n",
      "Train on 720000 samples, validate on 180000 samples\n",
      "Epoch 1/3\n",
      "720000/720000 - 126s - loss: 1.8533 - acc: 0.2704 - val_loss: 1.6541 - val_acc: 0.3382\n",
      "Epoch 2/3\n",
      "720000/720000 - 127s - loss: 1.5163 - acc: 0.4081 - val_loss: 1.5961 - val_acc: 0.3826\n",
      "Epoch 3/3\n",
      "720000/720000 - 126s - loss: 1.3158 - acc: 0.5179 - val_loss: 1.7247 - val_acc: 0.3710\n",
      "<tensorflow.python.keras.callbacks.History object at 0x7f66a62b48d0>\n",
      "Train on 720000 samples, validate on 180000 samples\n",
      "Epoch 1/3\n",
      "720000/720000 - 124s - loss: 1.8342 - acc: 0.2802 - val_loss: 1.6334 - val_acc: 0.3441\n",
      "Epoch 2/3\n",
      "720000/720000 - 127s - loss: 1.5023 - acc: 0.4144 - val_loss: 1.6061 - val_acc: 0.3830\n",
      "Epoch 3/3\n",
      "720000/720000 - 124s - loss: 1.3037 - acc: 0.5251 - val_loss: 1.7450 - val_acc: 0.3681\n",
      "<tensorflow.python.keras.callbacks.History object at 0x7f66a4012110>\n",
      "Train on 720000 samples, validate on 180000 samples\n",
      "Epoch 1/3\n",
      "720000/720000 - 125s - loss: 1.8279 - acc: 0.2855 - val_loss: 1.6853 - val_acc: 0.3312\n",
      "Epoch 2/3\n",
      "720000/720000 - 120s - loss: 1.5410 - acc: 0.3959 - val_loss: 1.6062 - val_acc: 0.3791\n",
      "Epoch 3/3\n",
      "720000/720000 - 121s - loss: 1.3441 - acc: 0.5028 - val_loss: 1.7177 - val_acc: 0.3710\n",
      "<tensorflow.python.keras.callbacks.History object at 0x7f6694ae7810>\n",
      "Train on 720000 samples, validate on 180000 samples\n",
      "Epoch 1/3\n",
      "720000/720000 - 120s - loss: 0.3300 - acc: 0.8618 - val_loss: 0.2463 - val_acc: 0.9161\n",
      "Epoch 2/3\n",
      "720000/720000 - 121s - loss: 0.1916 - acc: 0.9461 - val_loss: 0.2704 - val_acc: 0.9098\n",
      "Epoch 3/3\n",
      "720000/720000 - 121s - loss: 0.1499 - acc: 0.9662 - val_loss: 0.3101 - val_acc: 0.9032\n",
      "<tensorflow.python.keras.callbacks.History object at 0x7f6694bb2150>\n",
      "Train on 720000 samples, validate on 180000 samples\n",
      "Epoch 1/3\n",
      "720000/720000 - 123s - loss: 0.3242 - acc: 0.8655 - val_loss: 0.2460 - val_acc: 0.9173\n",
      "Epoch 2/3\n",
      "720000/720000 - 124s - loss: 0.1908 - acc: 0.9462 - val_loss: 0.2697 - val_acc: 0.9089\n",
      "Epoch 3/3\n",
      "720000/720000 - 124s - loss: 0.1515 - acc: 0.9656 - val_loss: 0.3062 - val_acc: 0.9021\n",
      "<tensorflow.python.keras.callbacks.History object at 0x7f66a47e83d0>\n",
      "Train on 720000 samples, validate on 180000 samples\n",
      "Epoch 1/3\n",
      "720000/720000 - 154s - loss: 0.3221 - acc: 0.8664 - val_loss: 0.2465 - val_acc: 0.9160\n",
      "Epoch 2/3\n",
      "720000/720000 - 149s - loss: 0.1884 - acc: 0.9470 - val_loss: 0.2719 - val_acc: 0.9096\n",
      "Epoch 3/3\n",
      "720000/720000 - 149s - loss: 0.1474 - acc: 0.9669 - val_loss: 0.3119 - val_acc: 0.9037\n",
      "<tensorflow.python.keras.callbacks.History object at 0x7f668c0e0210>\n",
      "3     272906\n",
      "4     153407\n",
      "2     147314\n",
      "5     144038\n",
      "6     135960\n",
      "7      78541\n",
      "8      23788\n",
      "1      19316\n",
      "9      15537\n",
      "10      9193\n",
      "Name: predicted_age, dtype: int64\n",
      "1    675932\n",
      "2    324068\n",
      "Name: predicted_gender, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#!/usr/bin/env python \n",
    "# encoding: utf-8 \n",
    "\n",
    "\"\"\"\n",
    "@version: v1.0\n",
    "@author: zhenglinghan\n",
    "@contact: 422807471@qq.com\n",
    "@software: PyCharm\n",
    "@file: baseline.py\n",
    "@time: 2020/5/7 22:23\n",
    "\"\"\"\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from tensorflow.python.keras.preprocessing.sequence import pad_sequences\n",
    "from deepctr.inputs import  DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names\n",
    "from deepctr.models import DeepFM\n",
    "from xdeepfm import xDeepFM\n",
    "import tensorflow as tf\n",
    "\n",
    "import os\n",
    "import gc\n",
    "import datetime as dt\n",
    "import warnings\n",
    "import joblib\n",
    "from gensim.models import Word2Vec\n",
    "import sys\n",
    "from tqdm import tqdm\n",
    "import time\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', None)\n",
    "pd.set_option('precision', 5)\n",
    "pd.set_option('display.float_format', lambda x: '%.5f' % x)\n",
    "pd.set_option('max_colwidth', 200)\n",
    "pd.set_option('display.width', 5000)\n",
    "\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.metrics import roc_curve, roc_auc_score, f1_score , accuracy_score\n",
    "from sklearn.preprocessing import LabelEncoder, MinMaxScaler\n",
    "    \n",
    "from catboost import CatBoostClassifier\n",
    "path_train = '../data/train_preliminary/'\n",
    "path_test = '../data/test/'\n",
    "\n",
    "\n",
    "\n",
    "# log\n",
    "class Logger(object):\n",
    "    def __init__(self, fileN=\"Default.log\"):\n",
    "        self.terminal = sys.stdout\n",
    "        self.log = open(fileN, \"a\",encoding='utf-8')\n",
    "\n",
    "    def write(self, message):\n",
    "        self.terminal.write(message)\n",
    "        self.log.write(message)\n",
    "\n",
    "    def flush(self):\n",
    "        pass\n",
    "    \n",
    "\n",
    "def reduce_mem_usage(df,features, verbose=True):\n",
    "    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n",
    "    start_mem = df.memory_usage().sum() / 1024 ** 2\n",
    "    for col in tqdm(features):\n",
    "        col_type = df[col].dtypes\n",
    "        if col_type in numerics:\n",
    "            c_min = df[col].min()\n",
    "            c_max = df[col].max()\n",
    "            if str(col_type)[:3] == 'int':\n",
    "                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n",
    "                    df[col] = df[col].astype(np.int8)\n",
    "                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n",
    "                    df[col] = df[col].astype(np.int16)\n",
    "                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n",
    "                    df[col] = df[col].astype(np.int32)\n",
    "                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n",
    "                    df[col] = df[col].astype(np.int64)\n",
    "            else:\n",
    "                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n",
    "                    df[col] = df[col].astype(np.float16)\n",
    "                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n",
    "                    df[col] = df[col].astype(np.float32)\n",
    "                else:\n",
    "                    df[col] = df[col].astype(np.float64)\n",
    "    end_mem = df.memory_usage().sum() / 1024 ** 2\n",
    "    if verbose:\n",
    "        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))\n",
    "    return df\n",
    "\n",
    "def split_self(x):\n",
    "    key_ans = x .split(' ')\n",
    "    for key in key_ans:\n",
    "        if key not in key2index:\n",
    "            # Notice : input value 0 is a special \"padding\",so we do not use 0 to encode valid feature for sequence input\n",
    "            key2index[key] = len(key2index) + 1\n",
    "    return list(map(lambda x: key2index[x], key_ans))\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    \n",
    "    sys.stdout = Logger(\"log_deepfm.txt\")\n",
    "    \n",
    "    datatrain = pd.read_hdf(path_train+'data_train.h5',key='ad')\n",
    "    datatest = pd.read_hdf(path_test+'data_test.h5',key='ad')\n",
    "\n",
    "    datatrainlog = pd.read_hdf(path_train+'data_train.h5',key='click_log')\n",
    "    datatestlog = pd.read_hdf(path_test+'data_test.h5',key='click_log')\n",
    "\n",
    "    datatrainlabel = pd.read_hdf(path_train + 'data_train.h5', key='user')\n",
    "\n",
    "    # 2个模型\n",
    "    \n",
    "    datatestlabel = pd.DataFrame(list(datatestlog['user_id'].unique()), columns=['user_id'])\n",
    "    datalabel = pd.concat([datatrainlabel,datatestlabel],ignore_index=True)\n",
    "    \n",
    "    del datatrainlabel,datatestlabel\n",
    "    gc.collect()\n",
    "    \n",
    "    datalog = pd.read_hdf('datalog.h5')\n",
    "    datalabel = pd.read_hdf('datalabel0516_all.h5')# 里面特征很多 不过几个序列特征被我截断到200个了\n",
    "\n",
    "    print('features finish')\n",
    "    gc.collect()\n",
    "    # 建模\n",
    "    datalabel['age'] = datalabel['age']-1\n",
    "    datalabel['gender'] = datalabel['gender']-1\n",
    "    \n",
    "    sparse_features = ['click_times_max_max',\n",
    "'click_times_max_min',\n",
    "'click_times_max_sum',\n",
    "'click_times_sum_max',\n",
    "'click_times_sum_min','time_max', 'time_min']# 几个类别特征\n",
    "    dense_features = [i for i in datalabel.columns if i not in sparse_features +['user_id','age','gender',\n",
    "'creative_ids_list',\n",
    " 'ad_ids_list',\n",
    " 'product_categorys_list',\n",
    " 'advertiser_ids_list',\n",
    " 'product_ids_list',\n",
    " 'industrys_list',\n",
    " 'click_timess_list']]\n",
    "    \n",
    "    datalabel[sparse_features] = datalabel[sparse_features].fillna('-1')\n",
    "    datalabel[dense_features] = datalabel[dense_features].fillna(0)\n",
    "    \n",
    "    # 区分 train test\n",
    "    traindata = datalabel.loc[~datalabel['age'].isna()].copy()\n",
    "    testdata = datalabel.loc[datalabel['age'].isna()].copy().reset_index(drop=True)\n",
    "    \n",
    "    targets = ['age','gender']\n",
    "\n",
    "    # 1.Label Encoding for sparse features,and process sequence features\n",
    "    for feat in sparse_features:\n",
    "        lbe = LabelEncoder()\n",
    "        lbe.fit(datalabel[feat])\n",
    "        traindata[feat] = lbe.transform(traindata[feat])\n",
    "        testdata[feat] = lbe.transform(testdata[feat])\n",
    "#     mms = MinMaxScaler(feature_range=(0, 1))\n",
    "#     traindata[dense_features] = mms.fit_transform(traindata[dense_features])\n",
    "#     testdata[dense_features] = mms.transform(testdata[dense_features])# 处理了，但是没有放入模型\n",
    "    print('process 1 finish')\n",
    "    # 2.count #unique features for each sparse field and generate feature config for sequence feature\n",
    "\n",
    "    fixlen_feature_columns = [SparseFeat(feat, datalabel[feat].nunique(), embedding_dim=16)\n",
    "                              for feat in sparse_features]\n",
    "    \n",
    "    linear_feature_columns = fixlen_feature_columns.copy()\n",
    "    dnn_feature_columns = fixlen_feature_columns.copy()\n",
    "    \n",
    "    # 3.generate input data for model\n",
    "    model_input = {name: traindata[name] for name in sparse_features}  #  \n",
    "    model_input_test = {name: testdata[name] for name in sparse_features}  #  \n",
    "    # 只用一个多值特征\n",
    "    for var in ['creative_ids_list']:\n",
    "        # preprocess the sequence feature\n",
    "        key2index = {}\n",
    "        genres_list = list(map(split_self, traindata[var].values))\n",
    "        genres_length = np.array(list(map(len, genres_list)))\n",
    "        genres_list_test = list(map(split_self, testdata[var].values))\n",
    "        genres_length_test = np.array(list(map(len, genres_list_test)))\n",
    "        max_len = max([max(genres_length),max(genres_length_test)])\n",
    "        print('max_len:',max_len)\n",
    "        # Notice : padding=`post`\n",
    "        genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )\n",
    "        genres_list_test = pad_sequences(genres_list_test, maxlen=max_len, padding='post', )\n",
    "        use_weighted_sequence = False\n",
    "        if use_weighted_sequence:\n",
    "            varlen_feature_columns = [VarLenSparseFeat(SparseFeat(var,vocabulary_size=len(\n",
    "            key2index) + 1, embedding_dim=16), maxlen=max_len, combiner='mean',\n",
    "                                                   weight_name=var+'_weight')]  # Notice : value 0 is for padding for sequence input feature\n",
    "        else:\n",
    "            varlen_feature_columns = [VarLenSparseFeat(SparseFeat(var, vocabulary_size=len(\n",
    "            key2index) + 1, embedding_dim=16), maxlen=max_len, combiner='mean',\n",
    "                                                   weight_name=None)]  # Notice : value 0 is for padding for sequence input feature\n",
    "        \n",
    "        model_input[var] = genres_list\n",
    "#         model_input[var + \"_weight\"] = np.random.randn(traindata.shape[0], max_len, 1) # 有毒没用\n",
    "        model_input_test[var] = genres_list_test\n",
    "#         model_input_test[var + \"_weight\"] = np.random.randn(testdata.shape[0], max_len, 1) # 有毒没用\n",
    "        \n",
    "        linear_feature_columns += varlen_feature_columns\n",
    "        dnn_feature_columns += varlen_feature_columns\n",
    "\n",
    "    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns) \n",
    "    del datalabel\n",
    "    gc.collect()\n",
    "#     features = [i for i in datalabel.columns if i != 'age' and i!='gender' and i!='user_id']\n",
    "    fold_num = 3\n",
    "    random_states = [ 0]  # 0, # 1111,1024,2019,2020\n",
    "    models = []\n",
    "    train_ccuracy_score = []\n",
    "    val_ccuracy_score = []\n",
    "    verbose = True\n",
    "    print('modeling!')\n",
    "\n",
    "    for label in targets:\n",
    "        pred_y = np.zeros((len(testdata),traindata[label].nunique()))\n",
    "        for i in range(len(random_states)):\n",
    "            for index in range(fold_num):\n",
    "                model = xDeepFM(linear_feature_columns, \n",
    "                dnn_feature_columns, \n",
    "                task='multiclass',\n",
    "                dnn_hidden_units=(64, 64, 64),\n",
    "                cin_layer_size=(64,64,64),\n",
    "                cin_activation='relu', \n",
    "                l2_reg_linear=0.1,\n",
    "#                 l2_reg_embedding=0.05, # 有毒参数\n",
    "#                 l2_reg_dnn=0.05,  # 有毒参数\n",
    "#                 l2_reg_cin=0.05,  # 有毒参数\n",
    "#                 init_std=0.01,  # 有毒参数\n",
    "                seed=2020, \n",
    "#                 dnn_dropout=0.5,\n",
    "                dnn_activation='relu', \n",
    "                num_class=traindata[label].nunique())\n",
    "                model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=\"sparse_categorical_crossentropy\", metrics=['acc'], )\n",
    "                \n",
    "                history = model.fit(model_input, traindata[label].values,\n",
    "                        batch_size=2048, epochs=3, verbose=2, validation_split=0.2, )\n",
    "                if verbose:\n",
    "                    print(history)\n",
    "                model.save('xdeepfm_model_base_{}.h5'.format(index))\n",
    "                # model = keras.models.load_model('xdeepfm_model_{}.h5'.format(index))\n",
    "                y_test_prob  = model.predict(model_input_test)               \n",
    "                pred_y += y_test_prob /fold_num/len(random_states)\n",
    "        testdata['predicted_{}'.format(label)] = np.argmax( pred_y,axis=1)+1\n",
    "    subs = testdata[['user_id','predicted_age','predicted_gender']]\n",
    "    print(subs['predicted_age'].value_counts())\n",
    "    print(subs['predicted_gender'].value_counts())\n",
    "    subs.to_csv('submission_basefeaturexdeepfm_base.csv',index=False,encoding='utf-8')\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}