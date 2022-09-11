from matplotlib.pyplot import hist
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
    model.add(LSTM(units=50,return_sequences=True,input_shape=(input_dim, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
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
    T = 24
    N_train = 300

    # build model
    if opts.model == 'MLP':
        model = MLP(T,T)
    elif opts.model == 'RNN':
        model = RNNmodel(T,T)
    
    # build data
    for i in range(10):
        df_dp = np.load(f"dataset/data_N365_{i}.npz")
        df_price = df_dp["price"]

        P1 = df_dp["p"].max()
        P2 = df_dp["d"].max()
        d = df_dp["d"]
        p = df_dp["p"]
        
        df_y= p-d
        test_price = df_price[:10]
        test_dr = df_y[:10]
        train_price = df_price[10:10+N_train]
        train_dr = df_y[10:10+N_train]
        
        history = model.fit(train_price,train_dr, validation_data = (test_price, test_dr), epochs=500, batch_size=128, verbose=0)
        L = history.history['loss']
        val_L = history.history['val_loss']
        np.savez(f'result/baseline/{opts.model}_loss_{i}.npz', loss=L, val_loss=val_L)
        if i == 0:
            model.save(f"result/baseline/model_{opts.model}")
