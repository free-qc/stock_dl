# -*- coding: utf-8 -*-
"""
Created on 2018/7/26
only kline image(CNN)
holding period 5, 10, 20,模型更新跟随holding period
对比模型，SVR，LR
评价指标 Total Return，Daily Mean Return，Max Drawdown
根据回归结果，选取每日前k支股票进行持有，k 5,10,20,50

@author: Free_QC
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from lib.model import CNN_model
from lib.utils import generate_data
from lib.image_generator import generate_ta_imgs, generate_kline_imgs
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    bath_size = 20
    nb_epoch = 20
    update_period = 20
    holding_period_list = [5, 10, 20]
    # top_k_list = [5, 10, 20, 50]
    # 16，17所有交易日
    all_trade_days = pd.read_csv('../data/trade_days.csv', parse_dates=[1], index_col=0, header=None)
    input_dir = '../input/kline'
    output_dir = '../output/kline'
    stock_list = [f.split('.')[0] for f in os.listdir('../data/stock') if not f.startswith('.')]
    for i, holding_period in enumerate(holding_period_list):
        if not os.path.exists(output_dir + '/' + str(holding_period)):
            os.makedirs(output_dir + '/' + str(holding_period))
        # image_save = True if i == 0 else False
        generate_kline_imgs(pred_steps=holding_period,
                            labelling_method='regression',
                            image_save=False,
                            split_by_year=False)
        trade_days = all_trade_days[1][::holding_period]
        trade_days = trade_days.reset_index(drop=True)
        # 16年为训练集起始点
        # 根据t日的k线图，训练并选股后，第二天开盘立即买入，假设前一天收盘价和第二天开盘价相同？
        # 持有n天，在t+n天收盘时卖出
        for stock in stock_list:
            stock_pred = {}
            stock_data = np.load('{0}/{1}/{2}.npz'.format(input_dir, stock, stock))
            imgs, fin_data, labels = stock_data['imgs'], stock_data['fin_data'], stock_data['labels']
            stock_trade_days = fin_data[:, 0]
            update_model = False
            for idx, date in enumerate(trade_days[:-1]):
                stock_pred[date] = {}
                print('training {0} ,date:{1}'.format(stock, date))
                split_idx_arr = np.where(stock_trade_days == date)[0]
                if len(split_idx_arr) == 0:
                    # 这一天没有交易
                    print('{0} no trade in {1}'.format(stock, date))
                    stock_pred[date]['pred'] = np.nan
                    stock_pred[date]['pred'] = np.nan
                    if idx * holding_period % update_period == 0:
                        update_model = True
                else:
                    split_idx = split_idx_arr[0]
                    # 判断卖出日期是否正确
                    try:
                        sold_date = trade_days[idx + 1]
                        stock_sold_date = stock_trade_days[split_idx + holding_period]
                    except:
                        print('卖出日期错误')
                        sold_date = 0
                        stock_sold_date = -1
                    if sold_date == stock_sold_date:
                        if idx * holding_period % update_period == 0 or update_model:
                            train_X = imgs[:split_idx]
                            train_Y = labels[:split_idx]
                            test_x = imgs[split_idx].reshape(1, 112, 112, 3)
                            test_y = labels[split_idx]
                            train_X = train_X.astype('float32') / 255
                            test_x = test_x.astype('float32') / 255
                            K.clear_session()
                            tf.reset_default_graph()
                            cnn_model = CNN_model(input_shape=(112, 112, 3), method='Regression')
                            cnn_model.fit(train_X, train_Y,
                                          batch_size=bath_size,
                                          epochs=nb_epoch,
                                          verbose=2)
                            pred_y = cnn_model.predict(test_x)
                            stock_pred[date]['pred'] = pred_y[0][0]
                            stock_pred[date]['true'] = test_y
                            update_model = False
                        else:
                            test_x = imgs[split_idx].reshape(1, 112, 112, 3)
                            test_y = labels[split_idx]
                            test_x = test_x.astype('float32') / 255
                            pred_y = cnn_model.predict(test_x)
                            stock_pred[date]['pred'] = pred_y[0][0]
                            stock_pred[date]['true'] = test_y
                    else:
                        stock_pred[date]['pred'] = np.nan
                        stock_pred[date]['pred'] = np.nan
                pd.DataFrame(stock_pred).to_csv('{0}/{1}/{2}.csv'.format(output_dir, holding_period, stock))
            # pd.DataFrame(stock_pred).to_csv('{0}/{1}/{2}.csv'.format(output_dir, holding_period, stock))

