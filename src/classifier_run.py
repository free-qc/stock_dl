# -*- coding: utf-8 -*-
"""
Created on 2018/6/26

@author: Free_QC
"""
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from model import CNN_model
from utils import generate_data, metrics_eval
from image_generator import generate_ta_imgs, generate_kline_imgs

if __name__ == '__main__':

    bath_size = 6
    nb_epoch = 20
    train_interval = 5

    input_dir = '../input'
    output_dir = '../output'
    stock_list = [f.split('.')[0] for f in os.listdir('../data/stock') if not f.startswith('.')]
    input_years = [str(year) for year in range(2008, 2018)]
    fluc_period_params = [3, 5, 10, 15, 20]
    # fluc_period_params_test = [20]
    fluc_range_params = [0.01, 0.015, 0.02, 0.025, 0.03]
    params_tup = [(i, j) for i in fluc_range_params for j in fluc_period_params]
    cnn_model = CNN_model(input_shape=(112, 112, 3))

    for i, param in enumerate(params_tup):
        fluc_range, fluc_period = param
        # generate_imgs(fluc_range, fluc_period)
        if i == 0:
            generate_kline_imgs(fluc_range, fluc_period, labelling_method='fluc', image_save=False,
                                img_shape=(112, 112))
        else:
            generate_kline_imgs(fluc_range, fluc_period, labelling_method='fluc', image_save=False)
        for stock in stock_list:
            stock_dir = input_dir + '/' + 'kline/' + stock
            stock_output_dir = output_dir + '/' + stock + '/' + str(fluc_period) + '_' + str(fluc_range)
            if not os.path.exists(stock_output_dir):
                os.makedirs(stock_output_dir)
            res_dic = {}
            for idx in range(len(input_years) - train_interval):
                train_begin_idx = idx
                train_end_idx = idx + train_interval - 1
                test_idx = train_end_idx + 1
                train_year_interval = input_years[train_begin_idx:train_end_idx + 1]
                test_year = input_years[test_idx]
                train_x, train_y, test_x, test_y, train_fin_data, test_fin_data = generate_data(stock_dir,
                                                                                                train_year_interval,
                                                                                                test_year)
                print('training:{0},train_interval:{1}-->{2},test_year:{3}'.format(stock, train_year_interval[0],
                                                                                   train_year_interval[-1],
                                                                                   test_year))
                # assert len(np.unique(train_y)) == 3, '少个类别'
                if not len(np.unique(train_y)) == 3:
                    continue
                train_y = to_categorical(train_y)

                # test_y = to_categorical(test_y)

                history = cnn_model.fit(train_x, train_y,
                                        batch_size=bath_size,
                                        epochs=nb_epoch,
                                        verbose=2)
                # SELL:0  BUY:1  HOLD:2
                pred_y = cnn_model.predict_classes(test_x)
                # pred_y = to_categorical(pred_y)

                res_dic[test_year] = {}
                res_dic[test_year]['metrics_eval'] = metrics_eval(pred_y, test_y)
                # res_dic[test_year]['fin_eval'] = fin_eval(pred_y, train_fin_data, test_fin_data)
                score_df = pd.DataFrame(res_dic[test_year]['metrics_eval']['score'])
                score_df.to_csv(stock_output_dir + '/' + str(test_year) + '_score.csv')
                confusion_matrix_df = pd.DataFrame(res_dic[test_year]['metrics_eval']['confusion_matrix'],
                                                   index=[['Actual'] * 3, ['SELL', 'BUY', 'HOLD']],
                                                   columns=[['Predicted'] * 3, ['SELL', 'BUY', 'HOLD']])
                confusion_matrix_df.to_csv(stock_output_dir + '/' + str(test_year) + '_confutsion_matrix.csv')
            # res_process(res_dic, ['metrics_eval'], stock_output_dir)
