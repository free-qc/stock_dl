# -*- coding: utf-8 -*-
"""
Created on 2018/7/13

@author: Free_QC
"""
import pandas as pd
import numpy as np

from collections import Counter


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
