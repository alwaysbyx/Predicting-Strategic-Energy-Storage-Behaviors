from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import tensorflow as tf
import pandas as pd
import argparse
import numpy as np

tf.random.set_seed(0)

def MLP(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model


def RNNmodel(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(units=output_dim,return_sequences=True,input_shape=(input_dim, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=output_dim,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=output_dim,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_dim))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MLP", help="MLP or RNN")
    parser.add_argument("--save", type=int, default=0, help="whether to save")
    
    opts = parser.parse_args()
    T = 288
    N_test = 10
    N_train = 100

    # build model
    if opts.model == 'MLP':
        model = MLP(T,T)
    elif opts.model == 'RNN':
        model = RNNmodel(T,T)

    df_dp = np.load(f"dataset/UQ Tesla Battery - 2022 (1-min resolution).csv")
    df_price = df_dp['Energy Price ($/MWh)']
    df_y = df_dp['3 Phase Power (kW)']

    test_price = df_price[:N_train*T]
    test_dr = df_y[:N_train*T]
    train_price = df_price[N_train*T:(N_test+N_train)*T]
    train_dr = df_y[N_train*T:(N_test+N_train)*T]

    history = model.fit(train_price,train_dr, validation_data = (test_price, test_dr),  epochs=500, batch_size=128, verbose=0) #validation_data = (test_price, test_dr),
    L = history.history['loss']
    val_L = history.history['val_loss']
    np.savez(f'../result/baseline/{opts.model}_loss', loss=L, val_loss=val_L)
    model.save(f"../result/baseline/model_{opts.model}")