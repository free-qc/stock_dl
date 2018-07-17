# -*- coding: utf-8 -*-
"""
Created on 2018/6/26

@author: Free_QC
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils import to_categorical
from lib.model import CNN_model, CNN1D_model
from lib.utils import generate_data, metrics_eval
from lib.image_generator import generate_ta_imgs, generate_kline_imgs
import keras.backend.tensorflow_backend as KTF

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    bath_size = 20
    nb_epoch = 20
    train_interval = 5

    stock_list = [f.split('.')[0] for f in os.listdir('../data/stock') if not f.startswith('.')]
    input_years = [str(year) for year in range(2008, 2018)]
    pred_steps = [1, 3, 5, 10, 15, 20]
    fluc_range = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    window_size = [2, 3, 5, 7, 10]
    # pred_steps = [1]
    # fluc_range = [0.01]
    params = [(f_range, step, win_size) for f_range in fluc_range for step in pred_steps for win_size in window_size]
    for img_generator in [generate_ta_imgs]:
        generator_name = img_generator.__name__
        input_dir = '../input/ta' if generator_name == 'generate_ta_imgs' else '../input/kline'
        output_dir = input_dir.replace('input', 'output')
        img_input_shape = (15, 15) if generator_name == 'generate_ta_imgs' else (112, 112)
        # k steps predict
        for stock in stock_list:
            for f_range, step, win_size in params:
                if generator_name == 'generate_ta_imgs':
                    generate_ta_imgs(fluc_range=f_range, pred_steps=step, labelling_method='fluc')
                else:
                    generate_kline_imgs(fluc_range=f_range, pred_steps=step, labelling_method='fluc', image_save=False)
                stock_dir = input_dir + '/' + stock
                stock_output_dir = output_dir + '/' + stock + '/' + str(win_size) + '_' + str(f_range) + '_' + str(step)

                if not os.path.exists(stock_output_dir):
                    os.makedirs(stock_output_dir)
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
                    train_y = to_categorical(train_y)
                    if not train_y.shape[1] == 3:
                        continue
                    # clear session and build model
                    K.clear_session()
                    tf.reset_default_graph()
                    train_input_shape = train_x.shape
                    train_x = train_x.reshape(train_input_shape[0], 15, 15)
                    train_x = np.transpose(train_x, axes=[0, 2, 1])
                    test_input_shape = test_x.shape
                    test_x = test_x.reshape(test_input_shape[0], 15, 15)
                    test_x = np.transpose(test_x, axes=[0, 2, 1])

                    cnn1d_model = CNN1D_model(img_input_shape, window_size=win_size, method='Classification')
                    history = cnn1d_model.fit(train_x, train_y,
                                              batch_size=bath_size,
                                              epochs=nb_epoch,
                                              verbose=2
                                              )
                    # SELL:0  BUY:1  HOLD:2
                    pred_y = cnn1d_model.predict_classes(test_x)
                    # metrics
                    metrics = metrics_eval(test_y, pred_y)
                    score_df = pd.DataFrame.from_dict(metrics['score'], orient='index')
                    score_df.to_csv(stock_output_dir + '/' + str(test_year) + '_score.csv')
                    confusion_matrix_df = pd.DataFrame(metrics['confusion_matrix'],
                                                       index=[['Actual'] * 3, ['SELL', 'BUY', 'HOLD']],
                                                       columns=[['Predicted'] * 3, ['SELL', 'BUY', 'HOLD']])
                    confusion_matrix_df.to_csv(stock_output_dir + '/' + str(test_year) + '_confutsion_matrix.csv')
