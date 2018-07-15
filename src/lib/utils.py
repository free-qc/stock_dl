# -*- coding: utf-8 -*-
"""
Created on 2018/7/13

@author: Free_QC
"""
import pandas as pd
import numpy as np

from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from lib.model import MLP_model, LSTM_model


def generate_data(stock_dir, train_interval, test_year):
    for year_idx, train_year in enumerate(train_interval):
        train_arrs = np.load(stock_dir + '/' + train_year + '.npz')
        if year_idx == 0:
            train_X = train_arrs['imgs']
            train_Y = train_arrs['labels']
            train_fin_data = train_arrs['fin_data']
        else:
            train_X = np.vstack((train_X, train_arrs['imgs']))
            train_Y = np.hstack((train_Y, train_arrs['labels']))
            train_fin_data = np.vstack((train_fin_data, train_arrs['fin_data']))
    test_arrs = np.load(stock_dir + '/' + test_year + '.npz')
    test_X = test_arrs['imgs']
    test_Y = test_arrs['labels']
    test_fin_data = test_arrs['fin_data']
    if train_X.dtype == np.dtype('uint8'):
        train_X = train_X.astype('float32') / 255
        test_X = test_X.astype('float32') / 255
    return train_X, train_Y, test_X, test_Y, train_fin_data, test_fin_data


def metrics_eval(y_true, y_pred):
    score_funcs = (recall_score, precision_score, f1_score)
    y_true_shape = y_true.shape
    y_pred_shape = y_pred.shape
    y_true = output_reshape(y_true, y_true_shape)
    y_pred = output_reshape(y_pred, y_pred_shape)
    confusion_mat = confusion_matrix(y_true, y_pred)
    # SELL:0  BUY:1  HOLD:2
    score_dic = {'SELL': {}, 'BUY': {}, 'HOLD': {}}
    for func in score_funcs:
        score_res = func(y_true, y_pred, average=None)
        for idx, label in enumerate(score_dic):
            score_dic[label][func.__name__] = score_res[idx]
    return {'confusion_matrix': confusion_mat,
            'score': score_dic
            }


def output_reshape(arr, arr_shape):
    label_enc_factor = np.array([0, 1, 2], dtype='int8')
    if not len(arr_shape) == 1:
        if arr_shape[1] == 2:
            arr = np.hstack((arr, np.zeros((arr_shape[0], 1), dtype='int8')))
        assert arr_shape > 1, '只有一类？'
        arr = (arr * label_enc_factor).sum(axis=1)
    return arr


def finan_eval(y_pred, train_fin_data, test_fin_data):
    # fin_data: cols: date open close RSI14 SMA50(compared model)
    label_enc_factor = np.array([0, 1, 2], dtype='int32')
    data_len = y_pred.shape[0]
    test_close_price = test_fin_data[:, 2]
    train_close_price = train_fin_data[:, 2]
    # CNN-TA
    if not len(y_pred.shape) == 1:
        y_pred = (y_pred * label_enc_factor).sum(axis=1)
    CNNTA_y_pred = y_pred
    CNNTA_ar = ar_process(CNNTA_y_pred, test_fin_data)
    # buy and hold
    BAH_y_pred = np.array([1] + [2] * (data_len - 2) + [0], dtype='int8')
    BAH_ar = ar_process(BAH_y_pred, test_fin_data)
    # RSI
    RSI14 = test_fin_data[:, 3]
    RSI_y_pred = []
    for i in RSI14:
        if i > 70:
            RSI_y_pred += [0]
        elif i < 30:
            RSI_y_pred += [1]
        else:
            RSI_y_pred += [2]
    RSI14_ar = ar_process(RSI_y_pred, test_fin_data)
    # SMA50
    SMA50 = test_fin_data[:, 4]
    SMA50_y_pred = test_close_price > SMA50
    SMA50_ar = ar_process(SMA50_y_pred, test_fin_data)
    # LSTM
    # LSTM_y_pred = LSTM_model(train_close_price, test_close_price)
    # LSTM_ar = ar_process(LSTM_y_pred, test_fin_data)
    # MLP
    MLP_y_pred = MLP_model(train_close_price, test_close_price)
    MLP_ar = ar_process(MLP_y_pred, test_fin_data)
    return {'CNNTA': CNNTA_ar,
            'BAH': BAH_ar,
            'RSI': RSI14_ar,
            'SMA': SMA50_ar,
            # 'LSTM': LSTM_ar,
            'MLP': MLP_ar}


def ar_process(y_pred, fin_data):
    # fin_data: cols: date open close RSI14 SMA50(compared model)
    # 用 close_price
    # SELL:0  BUY:1  HOLD:2
    total_money = 10000
    trans_count = 0
    num_stocks = 0
    hold_flag = 0
    for idx, label in enumerate(y_pred):
        if (label == 1) & (hold_flag == 0):
            num_stocks = total_money / fin_data[idx, 2]
            trans_count += 1
            hold_flag = 1
        if (label == 0) & (hold_flag == 1):
            total_money = num_stocks * fin_data[idx, 2]
            trans_count += 1
            hold_flag = 0
    total_money = total_money - trans_count
    ar = (total_money / 10000 - 1) * 100
    return {'ar': ar,
            'total_money': total_money,
            'trans_count': trans_count}


def res_process(res_dic, eval_list, output_dir):
    final_res = {}
    for eval in eval_list:
        for i, year in enumerate(res_dic):
            year_res = res_dic[year]
            eval_res = year_res[eval]
            if eval == 'metrics_eval':
                # score_dic = {'SELL': {}, 'BUY': {}, 'HOLD': {}}
                # for func in score_funcs:
                #     score_res = func(y_true, y_pred, average=None)
                #     for idx, label in enumerate(score_dic):
                #         score_dic[label][func.__name__] = score_res[idx]
                # return {'confusion_matrix': confusion_mat,
                #         'score': score_dic
                #         }
                # precision_score, recall_score, f1_score
                if i == 0:
                    final_res[eval] = {}
                    final_res[eval]['confusion_matrix'] = eval_res['confusion_matrix']
                    final_res[eval]['score'] = eval_res['score']
                else:
                    final_res[eval]['confusion_matrix'] += eval_res['confusion_matrix']
                    for label in ['SELL', 'BUY', 'HOLD']:
                        final_res[eval]['score'][label] = dict(
                            Counter(final_res[eval]['score'][label]) + Counter(eval_res['score'][label]))
            else:
                # return {'CNNTA': CNNTA_ar,
                #         'BAH': BAH_ar,
                #         'RSI': RSI14_ar,
                #         'SMA': SMA50_ar,
                #         # 'LSTM': LSTM_ar,
                #         'MLP': MLP_ar}
                if i == 0:
                    final_res[eval] = eval_res
                else:
                    final_res[eval] = dict(Counter(final_res[year][eval]) + Counter(eval_res))
    confusion_matrix_df = None
    score_df = None
    finan_se = None
    for eval in eval_list:
        if eval == 'metrics_eval':
            confusion_matrix_df = pd.DataFrame(final_res[eval]['confusion_matrix'],
                                               index=[['Actual'] * 3, ['SELL', 'HOLD', 'BUY']],
                                               columns=[['Predicted'] * 3, ['SELL', 'HOLD', 'BUY']])
            score_df = pd.DataFrame(final_res[eval]['score'])
        else:
            finan_se = pd.Series(final_res[eval])
    try:
        year_count = len(res_dic)
        score_df = score_df / year_count
        finan_se = finan_se / year_count
    except:
        pass
    try:
        confusion_matrix_df.to_csv(output_dir + '/' + 'confusion_matrix.csv')
        score_df.to_csv(output_dir + '/' + 'score.csv')
        finan_se.to_csv(output_dir + '/' + 'finan_eval.csv')
    except:
        pass
    # with open(output_dir + '/' + 'res.json', 'w', encoding='utf-8') as f:
    #     json.dump(res_dic, f)
    return confusion_matrix_df, score_df, finan_se
