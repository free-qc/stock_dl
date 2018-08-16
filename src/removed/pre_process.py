# -*- coding: utf-8 -*-
"""
Created on 2018/6/26

@author: Free_QC
"""

import os
import pandas as pd

input_dir = './'
output_dir = './stock'
stocks_path = [f for f in os.listdir(input_dir) if not f.startswith('.')]

for stock in stocks_path:
    stock_df = pd.read_csv((input_dir + '/' + stock), header=0, encoding='gbk')
    cols = ['日期', '开盘价', '收盘价', '最高价', '最低价', '成交量']
    stock_df = stock_df.loc[:, cols]
    stock_df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
    stock_df.to_csv((output_dir + '/' + stock), index_label=False, encoding='utf-8')
