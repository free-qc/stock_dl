# -*- coding: utf-8 -*-
"""
Created on 2018/7/26

@author: Free_QC
"""

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from lib.model import MLP_model, LSTM_model


def metrics_eval(y_true, y_pred):
    score_funcs = (recall_score, precision_score, f1_score)
    y_true_shape = y_true.shape
    y_pred_shape = y_pred.shape
    y_true = output_reshape(y_true, y_true_shape)
    y_pred = output_reshape(y_pred, y_pred_shape)
    confusion_mat = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    # SELL:0  BUY:1  HOLD:2
    score_dic = {'SELL': {}, 'BUY': {}, 'HOLD': {}}
    for func in score_funcs:
        score_res = func(y_true, y_pred, average=None, labels=[0, 1, 2])
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
