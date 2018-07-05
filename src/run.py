# -*- coding: utf-8 -*-
"""
Created on 2018/6/26

@author: Free_QC
"""
import os

from cnnta import generate_imgs, generate_data, metrics_eval, finan_eval, res_process, CNN_model
from keras.utils import to_categorical

if __name__ == '__main__':
    generate_imgs(0.015)

    bath_size = 20
    nb_epoch = 2
    train_interval = 5

    cnn_model = CNN_model()

    input_dir = '../input'
    output_dir = '../output'
    stock_list = [f for f in os.listdir(input_dir) if not f.startswith('.')]
    input_years = [str(year) for year in range(2008, 2018)]
    for stock in stock_list:
        stock_dir = input_dir + '/' + stock
        stock_output_dir = output_dir + '/' + stock
        if not os.path.exists(stock_output_dir):
            os.mkdir(stock_output_dir)
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
                                                                               train_year_interval[-1], test_year))
            train_y = to_categorical(train_y)
            test_y = to_categorical(test_y)

            history = cnn_model.fit(train_x, train_y,
                                    batch_size=bath_size,
                                    epochs=nb_epoch,
                                    class_weight={0: 10, 1: 10, 2: 1},
                                    verbose=2)
            # SELL:0  BUY:1  HOLD:2
            pred_y = cnn_model.predict_classes(test_x)
            pred_y = to_categorical(pred_y)

            res_dic[test_year] = {}
            res_dic[test_year]['metrics_eval'] = metrics_eval(pred_y, test_y)
            # res_dic[test_year]['fin_eval'] = fin_eval(pred_y, train_fin_data, test_fin_data)
        res_process(res_dic, ['metrics_eval'], stock_output_dir)
        print(res_dic)
