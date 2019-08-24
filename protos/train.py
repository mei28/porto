import pandas as pd
import numpy as np
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
# 交差検証
from sklearn.metrics import log_loss, roc_auc_score
from load_data import load_train_data, load_test_data

SAMPLE_SUBMIT_FILE = '../input/sample_submission.csv'
logger = getLogger(__name__)
DIR = 'result_tmp/'
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

    all_params = {'C': [10 ** i for i in range(-1, 2)],
                  'fit_intercept': [True, False],
                  'penalty': ['l2', 'l1'],
                  'random_state': [0]
                  }

    min_score = 100
    min_params = None
    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params:{}'.format(params))

        list_logloss = []
        list_auc = []
        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            clf = LogisticRegression(**params)
            clf.fit(trn_x, trn_y)

            pred = clf.predict_proba(val_x)[:, 1]
            sc_logloss = log_loss(val_y, pred)
            sc_auc = - roc_auc_score(val_y, pred)

            logger.debug('logloss:{}, auc:{}'.format(sc_logloss, sc_auc))
            list_logloss.append(sc_logloss)
            list_auc.append(sc_auc)

        sc_auc = np.mean(list_auc)
        sc_logloss = np.mean(list_logloss)


        if min_score > sc_auc:
            min_score = sc_auc
            min_params = params

        logger.info('logloss:{}, auc:{}'.format(np.mean(list_logloss), np.mean(list_auc)))
        logger.info('current min score: {}, params{}'.format(min_score, min_params))

    logger.info('minimum auc: {}'.format(min_score))
    logger.info('minimum params: {}'.format(min_params))

    logger.info('train end')

    clf = LogisticRegression(**min_params)
    clf.fit(x_train,y_train)
    df = load_test_data()

    x_test = df[use_cols]
    x_test = df[use_cols].sort_values('id')
    logger.info('test data load end. {}'.format(x_test.shape))
    pred_test = clf.predict_proba(x_test)[:, 1]

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
    df_submit['target'] = pred_test

    df_submit.to_csv(DIR + 'submit.csv', index=False)
    logger.info('end')
