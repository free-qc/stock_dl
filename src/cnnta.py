# -*- coding: utf-8 -*-
"""
Created on 2018/6/28

@author: Free_QC
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import Counter
from talib.abstract import *
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.finance import candlestick_ochl
from PIL import Image
from pandas import datetime


def generate_imgs(fluc_range=0.01, labelling_method='fluc'):
    print('=================generating imgs===============')
    stocks_dir = '../data/stock'
    stocks_names = [f for f in os.listdir(stocks_dir) if not f.startswith('.')]
    output_dir = '../input'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    indicators = [MACD, RSI, WILLR, SMA, EMA, WMA, HMA, TEMA, CCI, CMO, MACD, PPO, ROC, MFI, DX, SAR]
    for stock in stocks_names:
        # date open close high low volume
        input_df = pd.read_csv(stocks_dir + '/' + stock, parse_dates=[0], header=0)
        # 原来存储为图片，现在存储为.npz
        # stock_year_path = output_dir + '/' + stock + '/' + str(year)
        # for class_name in ['SELL', 'HOLD', 'BUY']:
        #     class_path = stock_year_path + '/' + class_name
        #     if not os.path.exists(class_path):
        #         os.makedirs(class_path)
        stock_dir = output_dir + '/' + stock.split('.')[0]
        if not os.path.exists(stock_dir):
            os.mkdir(stock_dir)
        indicators_df = ta_process(input_df, indicators, fluc_range, labelling_method)
        # 按年份分开
        year_se = indicators_df['date'].map(lambda x: x.year)
        for year in range(2008, 2018):
            year_df = indicators_df[year_se == year]
            # date open close high low volume label SMA50(compared model) indicators....
            indicators_arr = year_df.iloc[:, 8:].values
            arr_len = indicators_arr.shape[0]
            indicators_arr = indicators_arr.reshape(arr_len, 225)
            # 归一化到(0,1)
            max_val = indicators_arr.max(axis=1)
            min_val = indicators_arr.min(axis=1)
            indicators_arr = (indicators_arr - min_val[:, np.newaxis]) / (max_val - min_val)[:, np.newaxis]
            indicators_arr = indicators_arr.reshape(arr_len, 15, 15, 1)
            indicators_fin_data = year_df.loc[:, ['date',
                                                  'open',
                                                  'close',
                                                  'RSI14',
                                                  'SMA50']].values
            indicators_labels = year_df['label'].values.astype('int8')
            # 存储
            # imgs:图片(n,15,15,1)
            # fin_data:日期(n,),cols: date open close RSI14 SMA50(compared model)
            # labels:标签(n,)
            np.savez('{0}/{1}.npz'.format(stock_dir, year), imgs=indicators_arr, fin_data=indicators_fin_data,
                     labels=indicators_labels)
    print('=====================done======================')


def generate_kline_imgs(fluc_range=0.01, labelling_method='fluc', image_save=False):
    stocks_dir = '../data/stock'
    stocks_names = [f for f in os.listdir(stocks_dir) if not f.startswith('.')]
    output_dir = '../input/kline'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for stock in stocks_names:
        # date open close high low volume
        input_df = pd.read_csv(stocks_dir + '/' + stock, parse_dates=[0], header=0)
        input_df = input_df[input_df['date'] < datetime(2018, 1, 1)]
        stock_dir = output_dir + '/' + stock.split('.')[0]
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)
        labels_arr = labelling(input_df, fluc_range=fluc_range, method=labelling_method)
        # 按年份建立文件夹，以便存储img
        for year in range(2008, 2018):
            stock_year_dir = stock_dir + '/' + str(year)
            if not os.path.exists(stock_year_dir):
                os.mkdir(stock_year_dir)
        # 生成img并存储
        imgs_arr = kline_imgs(input_df, stock_dir, image_save)
        # imgs_arr不包含前19天,labels_arr也需要裁剪
        labels_arr = labels_arr[19:-1]
        fin_data = input_df.loc[:, ['date',
                                    'open',
                                    'close']]
        fin_data = fin_data.iloc[19:-1].values
        year_se = input_df['date'].apply(lambda x: x.year)
        year_se = year_se.iloc[19:-1]
        assert imgs_arr.shape[0] == labels_arr.shape[0] == fin_data.shape[0], "labels和imgs或fin_data长度不同"
        for year in range(2008, 2018):
            year_fin_data = fin_data[year_se == year]
            year_imgs_arr = imgs_arr[year_se == year]
            year_labels_arr = labels_arr[year_se == year]
            np.savez('{0}/{1}.npz'.format(stock_dir, year), imgs=year_imgs_arr, fin_data=year_fin_data,
                     labels=year_labels_arr)


def kline_imgs(input_df, stock_dir, image_save=False):
    price_df = input_df.loc[:, ['date', 'open', 'close', 'high', 'low']]
    date_se = input_df.loc[:, 'date'][19:]
    num_imgs = price_df.shape[0] - 20 + 1
    # TODO 相对位置
    # price_max = price_df.max()
    # price_min = price_df.min()
    plt.grid(False)
    # 用来储存imgs
    imgs_arr = np.zeros(shape=(num_imgs, 224, 224, 3))
    for num in range(num_imgs):
        begin_idx = num
        end_idx = num + 19
        date = str(date_se[end_idx])[:10]
        year = date[:4]
        img_path = stock_dir + '/' + year + '/' + date + '.png'
        img_df = price_df.iloc[begin_idx:end_idx + 1]
        # 图像不标注日期，为了使k线连续，改为连续的数字即可（int或float都可以？）
        img_df['date'] = range(20)
        if image_save:
            # 开始作图
            fig, ax = plt.subplots()
            candlestick_ochl(ax, img_df.values, colorup='g', colordown='r')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.set_size_inches(2.24 / 3, 2.24 / 3)
            ax.set_facecolor('black')
            fig.set_facecolor('black')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0.1, 0.1)
            fig.savefig(img_path, format='png', facecolor='black', dpi=300, pad_inches=0)
            plt.close('all')
        imgs_arr[num] = np.array(Image.open(img_path).convert(mode='RGB'))
    # imgs_arr包含最后一天,最后一天没有label,去掉
    return imgs_arr[:-1]


def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(15, 15, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[categorical_accuracy])
    return model


def ta_process(input_df, indicators, fluc_range, labelling_method='fluc'):
    intervals = list(range(6, 21))
    input_df['label'] = labelling(input_df, fluc_range, method=labelling_method)
    input_df['SMA50'] = SMA(input_df, timeperiod=50)
    for indr in indicators:
        for intr in intervals:
            if indr == HMA:
                input_df['HMA' + str(intr)] = indr(input_df, timeperiod=intr)
            elif indr == MACD:
                input_df[indr.info['name'] + str(intr)] = indr(input_df, fastperiod=12,
                                                               slowperiod=26,
                                                               signalperiod=intr)['macdhist']
            else:
                input_df[indr.info['name'] + str(intr)] = indr(input_df, timeperiod=intr)
    input_df = input_df.dropna()
    input_df.reset_index(drop=True, inplace=True)
    # 技术指标的排序
    # oderings = [RSI, WILLR, SMA, EMA, WMA, HMA, TEMA, CCI, CMO, MACD, PPO, ROC, MFI, DX, SAR]
    # input_df = input_df.loc[:, oderings]
    return input_df


def labelling(input_df, fluc_range, method='fluc'):
    # 返回的label_arr长度均为输入数据长度
    if method == 'time_window':
        close_price_arr = input_df['close'].values
        arr_len = close_price_arr.shape[0]
        # 时间窗口是11，前10天缺失
        label_list = [np.nan] * 10
        for idx in range(10, arr_len):
            begin_idx = idx - 10
            end_idx = idx
            middle_idx = 5
            window_arr = close_price_arr[begin_idx:end_idx + 1]
            min_idx = window_arr.argmin()
            max_idx = window_arr.argmax()
            # SELL:0  BUY:1  HOLD:2
            if middle_idx == max_idx:
                label_list.append(0)
            elif middle_idx == min_idx:
                label_list.append(1)
            else:
                label_list.append(2)
        return np.array(label_list)
    if method == 'fluc':
        close_price_arr = input_df['close'].values
        close_arr_len = close_price_arr.shape[0]
        label_arr = np.zeros(shape=close_arr_len)
        label_arr[-1] = np.nan
        for idx in range(close_arr_len - 1):
            this_day_close = close_price_arr[idx]
            next_day_close = close_price_arr[idx + 1]
            close_diff = next_day_close - this_day_close
            # SELL:0  BUY:1  HOLD:2
            if close_diff < (-fluc_range):
                label_arr[idx] = 0
            elif close_diff > fluc_range:
                label_arr[idx] = 1
            else:
                label_arr[idx] = 2
        return label_arr
    if method == 'regression':
        label_arr = input_df['close'].values[1:]
        label_arr = np.append(label_arr, np.nan)
        return label_arr


def HMA(inputs, price='close', timeperiod=10):
    df = inputs.copy()
    X = 2 * WMA(df, price=price, timeperiod=timeperiod // 2) - WMA(df, price=price, timeperiod=timeperiod)
    HMA = WMA(df[(timeperiod - 1):], int(np.sqrt(timeperiod)))
    return pd.concat([X[:(timeperiod - 1)], HMA])


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


def LSTM_model(train_data, test_data, time_steps=240):
    train_x, test_x, train_y, test_y = generate_compared_data(train_data, test_data, time_steps=time_steps)

    LSTM_model = Sequential()
    LSTM_model.add(LSTM(25, input_shape=(time_steps, train_x.shape[2]), dropout=0.5))
    LSTM_model.add(Dense(1))
    LSTM_model.compile(loss='mean_squared_error', optimizer='adam')
    LSTM_model.fit(train_x, train_y, epochs=1000, batch_size=20, verbose=2)
    pred_y = LSTM_model.predict(test_x)
    return test_y > pred_y


def MLP_model(train_data, test_data, time_steps=100):
    train_x, test_x, train_y, test_y = generate_compared_data(train_data, test_data, time_steps=time_steps)
    train_x = train_x[:, :, 0]
    test_x = test_x[:, :, 0]

    MLP_model = Sequential()
    MLP_model.add(Dense(10, activation='relu', input_shape=(train_x.shape[1],)))
    MLP_model.add(Dense(5, activation='relu'))
    MLP_model.add(Dense(1, activation='relu'))
    MLP_model.compile(loss='mean_squared_error', optimizer='adam')

    MLP_model.fit(train_x, train_y,
                  batch_size=200,
                  epochs=20,
                  verbose=2,
                  )
    pred_y = MLP_model.predict(test_x)
    pred_y = pred_y.flatten()
    return test_y > pred_y


def generate_compared_data(train_data, test_data, time_steps=100):
    test_arr_len = test_data.shape[0]
    input_arr = np.concatenate((train_data, test_data))
    data_x = []
    data_y = []
    for i in range(len(input_arr) - time_steps):
        data_x.append(input_arr[i:(i + time_steps)])
        data_y.append(input_arr[i + time_steps])
    data_x = np.array(data_x)
    data_y = np.array(data_y).flatten()
    data_x = data_x[:, :, np.newaxis]
    # train test
    # TODO
    train_x, test_x = data_x[:-test_arr_len], data_x[-test_arr_len:]
    train_y, test_y = data_y[:-test_arr_len], data_y[-test_arr_len:]
    return train_x, test_x, train_y, test_y


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

# if __name__ == '__main__':
#     generate_kline_imgs()
#     from keras.utils import to_categorical
#     from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
#     from keras.metrics import categorical_accuracy
#
#     bath_size = 20
#     nb_epoch = 200
#     train_interval = 5
#
#     model = Sequential()
#     model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(15, 15, 1), activation='relu'))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(rate=0.25))
#     model.add(Flatten())
#     model.add(Dense(units=128, activation='relu'))
#     model.add(Dropout(rate=0.5))
#     model.add(Dense(units=3, activation='softmax'))
#
#     input_dir = '../input'
#     train_x, train_y, test_x, test_y, train_fin_data, test_fin_data = generate_data(input_dir + '/' + 'SH600000',
#                                                                                     ['2008', '2009', '2010', '2011',
#                                                                                      '2012'],
#                                                                                     '2013')
#
#     input_years = [str(year) for year in range(2008, 2019)]
#
#     train_y = to_categorical(train_y)
#     test_y = to_categorical(test_y)
#     model.compile(loss='categorical_crossentropy', optimizer='adam',
#                   metrics=[categorical_accuracy])
#     history = model.fit(train_x, train_y,
#                         batch_size=bath_size,
#                         epochs=nb_epoch,
#                         class_weight={0: 10, 1: 10, 2: 1},
#                         verbose=2)
#     pred_y = model.predict_classes(test_x)
#     print(metrics_eval(pred_y, test_y))
