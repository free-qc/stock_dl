# -*- coding: utf-8 -*-
"""
Created on 2018/7/13

@author: Free_QC
"""
import os
import pandas as pd
import tensorflow as tf
from keras import backend as K
from lib.model import CNN_model
from lib.utils import generate_data
from lib.image_generator import generate_ta_imgs, generate_kline_imgs
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    bath_size = 20
    nb_epoch = 20
    train_interval = 5

    stock_list = [f.split('.')[0] for f in os.listdir('../data/stock') if not f.startswith('.')]
    input_years = [str(year) for year in range(2008, 2018)]
    pred_steps = [1, 3, 5, 10, 15, 20]
    for img_generator in [generate_kline_imgs]:
        generator_name = img_generator.__name__
        input_dir = '../input/ta' if generator_name == 'generate_ta_imgs' else '../input/kline'
        output_dir = input_dir.replace('input', 'output/reg')
        img_input_shape = (15, 15, 1) if generator_name == 'generate_ta_imgs' else (112, 112, 3)
        # k steps predict
        for i, step in enumerate(pred_steps[5:6]):
            if generator_name == 'generate_ta_imgs':
                generate_ta_imgs(labelling_method='regression', pred_steps=step)
            else:
                generate_kline_imgs(pred_steps=step, labelling_method='regression', image_save=False)
            for stock in stock_list:
                stock_dir = input_dir + '/' + stock
                stock_output_dir = output_dir + '/' + stock
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
                    # clear session and build model
                    K.clear_session()
                    tf.reset_default_graph()
                    cnn_model = CNN_model(input_shape=img_input_shape, method='Regression')
                    history = cnn_model.fit(train_x, train_y,
                                            batch_size=bath_size,
                                            epochs=nb_epoch,
                                            verbose=2)
                    # SELL:0  BUY:1  HOLD:2
                    pred_y = cnn_model.predict(test_x)
                    # mean_absolute_error, mean_squared_error
                    res_dic[test_year] = {}
                    res_dic[test_year]['mean_absolute_error'] = mean_absolute_error(test_y, pred_y)
                    res_dic[test_year]['mean_squared_error'] = mean_squared_error(test_y, pred_y)
                pd.DataFrame.from_dict(res_dic, orient='index').to_csv(stock_output_dir + '/' + str(step) + '.csv')
