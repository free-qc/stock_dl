# -*- coding: utf-8 -*-
"""
Created on 2018/7/13

@author: Free_QC
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from progress_bar import ProgressBar
from talib.abstract import *
from matplotlib.finance import candlestick_ochl
from PIL import Image

np.seterr(invalid='ignore')


def generate_ta_imgs(fluc_range=0.01, fluc_period=5, labelling_method='fluc', pred_steps=1):
    print('=================generating imgs===============')
    stocks_dir = '../data/stock'
    stocks_names = [f for f in os.listdir(stocks_dir) if not f.startswith('.')]
    output_dir = '../input/ta'
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
        indicators_df = ta_process(input_df, indicators, fluc_range, fluc_period, labelling_method, pred_steps)
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


def generate_kline_imgs(fluc_range=0.01, fluc_period=5, pred_steps=1, labelling_method='fluc', image_save=False,
                        img_shape=(112, 112)):
    stocks_dir = '../data/stock'
    stocks_names = [f for f in os.listdir(stocks_dir) if not f.startswith('.')]
    output_dir = '../input/kline'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for stock in stocks_names:
        # date open close high low volume
        input_df = pd.read_csv(stocks_dir + '/' + stock, parse_dates=[0], header=0)
        stock_dir = output_dir + '/' + stock.split('.')[0]
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)
        labels_arr = labelling(input_df, fluc_range=fluc_range, fluc_period=fluc_period, method=labelling_method,
                               pred_steps=pred_steps)
        # 按年份建立文件夹，以便存储img
        for year in range(2008, 2019):
            stock_year_dir = stock_dir + '/' + str(year)
            if not os.path.exists(stock_year_dir):
                os.mkdir(stock_year_dir)
        # 生成img并存储
        if image_save:
            imgs_arr = kline_imgs(input_df, stock_dir, image_save, img_shape)
            imgs_arr = imgs_arr[:-fluc_period]
        # imgs_arr不包含前19天,labels_arr也需要裁剪
        labels_arr = labels_arr[19:-fluc_period].astype('uint8')
        fin_data = input_df.loc[:, ['date',
                                    'open',
                                    'close']]
        fin_data = fin_data.iloc[19:-fluc_period].values
        year_se = input_df['date'].apply(lambda x: x.year)
        year_se = year_se.iloc[19:-fluc_period]
        # assert imgs_arr.shape[0] == labels_arr.shape[0] == fin_data.shape[0], "labels和imgs或fin_data长度不同"
        for year in range(2008, 2018):
            if not image_save:
                year_imgs_arr = np.load('{0}/{1}.npz'.format(stock_dir, year))['imgs']
            else:
                year_imgs_arr = imgs_arr[year_se == year]
            year_fin_data = fin_data[year_se == year]
            year_labels_arr = labels_arr[year_se == year]
            np.savez('{0}/{1}.npz'.format(stock_dir, year), imgs=year_imgs_arr, fin_data=year_fin_data,
                     labels=year_labels_arr)


def kline_imgs(input_df, stock_dir, image_save=False, img_shape=(112, 112)):
    price_df = input_df.loc[:, ['date', 'open', 'close', 'high', 'low']]
    date_se = input_df.loc[:, 'date'][19:]
    num_imgs = price_df.shape[0] - 20 + 1
    # TODO 相对位置
    # price_max = price_df.max()
    # price_min = price_df.min()
    plt.grid(False)
    # 用来储存imgs
    row_pix, col_pix = img_shape
    imgs_arr = np.zeros(shape=(num_imgs, row_pix, col_pix, 3), dtype='uint8')
    bar = ProgressBar(total=num_imgs)
    for num in range(num_imgs):
        begin_idx = num
        end_idx = num + 19
        date = str(date_se[end_idx])[:10]
        year = date[:4]
        img_path = stock_dir + '/' + year + '/' + date + '.png'
        img_df = pd.DataFrame(price_df.iloc[begin_idx:end_idx + 1])
        # 图像不标注日期，为了使k线连续，改为连续的数字即可（int或float都可以？）
        img_df.loc[:, 'date'] = list(range(20))
        if image_save:
            # 开始作图
            fig, ax = plt.subplots()
            candlestick_ochl(ax, img_df.values, width=0.005, colorup='g', colordown='r')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.set_size_inches(row_pix / 300, col_pix / 300)
            ax.set_facecolor('black')
            fig.set_facecolor('black')
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0.1, 0.1)
            fig.savefig(img_path, format='png', facecolor='black', dpi=300, pad_inches=0)
            plt.close('all')
        imgs_arr[num] = np.array(Image.open(img_path).convert(mode='RGB'))
        bar.move()
        bar.log('generating kline imgs')
    # imgs_arr不包含前19天
    return imgs_arr


def ta_process(input_df, indicators, fluc_range, fluc_period, labelling_method='fluc', pred_steps=1):
    intervals = list(range(6, 21))
    input_df['label'] = labelling(input_df, fluc_range, fluc_period, labelling_method, pred_steps)
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


def labelling(input_df, fluc_range=0.01, fluc_period=5, method='fluc', pred_steps=1):
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
        fluc_period = int(fluc_period)
        close_price_arr = input_df['close'].values
        close_arr_len = close_price_arr.shape[0]
        label_arr = np.zeros(shape=close_arr_len)
        period_log = np.log(close_price_arr / np.roll(close_price_arr, fluc_period))[fluc_period:]
        period_log = np.append(period_log, np.array([np.nan] * fluc_period))
        label_arr[period_log < -fluc_range] = 0
        label_arr[period_log > fluc_range] = 1
        label_arr[(-fluc_range < period_log) * (period_log < fluc_range)] = 2
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
