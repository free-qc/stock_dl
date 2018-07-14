# -*- coding: utf-8 -*-
"""
Created on 2018/7/13

@author: Free_QC
"""

from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def CNN_model(input_shape=(15, 15, 1)):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
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


def Kline_model(input_shape=(112, 112, 3)):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[categorical_accuracy])
    return model


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
