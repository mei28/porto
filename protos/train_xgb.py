import os
import sys

sys.path.append("")
import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
# 交差検証
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
from load_data import load_train_data, load_test_data
import xgboost as xgb

SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
logger = getLogger(__name__)
DIR = 'result_tmp/'


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g


def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', - gini(y, pred)


if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()
    x_train = df.drop('target', axis=1)
    y_train = df['target'].values
    use_cols = x_train.columns.values

    logger.debug('train columns: {} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    all_params = {'max_depth': [3, 5, 7],
                  'learning_rate': [0.1],
                  'min_child_weight': [3, 5, 10],
                  'n_estimators': [10000],
                  'colsample_bytree': [0.8, 0.9],
                  'colsample_bylevel': [0.8, 0.9],
                  'reg_alpha': [0, 0.1],
                  'max_delta_step': [0.1],
                  'seed': [0]
                  }

    min_score = 100
    min_params = None
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params:{}'.format(params))

        list_logloss = []
        list_auc = []
        list_gini = []
        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = xgb.sklearn.XGBClassifier(**params)
            clf.fit(trn_x,
                    trn_y,
                    eval_set=[(val_x, val_y)],
                    early_stopping_rounds=100,
                    eval_metric=gini_xgb
                    )

            pred = clf.predict_proba(val_x)[:, 1]
            sc_logloss = log_loss(val_y, pred)
            sc_auc = - roc_auc_score(val_y, pred)
            sc_gini = - gini(val_y, pred)

            logger.debug('logloss:{}, auc:{}, gini:{}'.format(sc_logloss, sc_auc, sc_gini))
            list_logloss.append(sc_logloss)
            list_auc.append(sc_auc)
            list_gini.append(sc_gini)

        sc_auc = np.mean(list_auc)
        sc_logloss = np.mean(list_logloss)
        sc_gini = np.mean(list_gini)

        # if min_score > sc_auc:
        #     min_score = sc_auc
        #     min_params = params
        if min_score > sc_gini:
            min_score = sc_gini
            min_params = params

        logger.info('logloss:{}, auc:{}, gini:{}'.format(np.mean(list_logloss), np.mean(list_auc), np.mean(list_gini)))
        logger.info('current min score: {}, params{}'.format(min_score, min_params))

    # logger.info('minimum auc: {}'.format(min_score))
    logger.info('minimum gini:{}'.format(min_score))
    logger.info('minimum params: {}'.format(min_params))

    logger.info('train end')

    clf = xgb.sklearn.XGBClassifier(**min_params)
    clf.fit(x_train, y_train)
    df = load_test_data()

    x_test = df[use_cols].sort_values('id')
    logger.info('test data load end. {}'.format(x_test.shape))
    pred_test = clf.predict_proba(x_test)[:, 1]

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)
    logger.info('end')
