import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
if __name__ == '__main__':

    label = np.argmax(
        np.load('label.npy', allow_pickle=True), axis=1).flatten()
    x_train = np.load('x_train.npy', allow_pickle=True)

    x_train, val_train, x_label, val_label = train_test_split(
        x_train, label, test_size=0.1, stratify=label)

    train_data = lgb.Dataset(x_train, label=x_label)
    validation_data = lgb.Dataset(val_train, label=val_label)

    params = {
        'objective': 'binary',
        # 'num_class': 2,
        'num_boost_round': 50000,
        'boosting_type': 'gbdt',
        'metric': {'auc'},
        'max_depth': 6,
        'num_leaves': 50,
        'learning_rate': 0.01,
        'bagging_freq': 5,
        'verbose': 1
    }
    gbm = lgb.train(params, train_data, valid_sets=[validation_data])

    x_test = np.load('x_test.npy', allow_pickle=True)
    ans_mtx = gbm.predict(x_test)
    np.save('ans_mtx.npy', ans_mtx)

    # ans_mtx = np.load('ans_mtx.npy', allow_pickle=True)
    # print(ans_mtx.min())
