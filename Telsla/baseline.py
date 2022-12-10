from matplotlib.pyplot import hist
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense, Bidirectional
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
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model


def RNNmodel(input_dim, output_dim):
    model = Sequential()
    # model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(input_dim, 1))) # work worse
    # model.add(Dense(output_dim))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(output_dim))
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
    T = 80
    N_test = 20
    N_train = 22

    # build model
    if opts.model == 'MLP':
        model = MLP(T,T)
    elif opts.model == 'RNN':
        model = RNNmodel(T,T)

    dataset = np.load(f"dataset/UQ-Tesla-2020.npz")
    df_price = dataset['price']
    df_y = dataset['dr']
    e0 = dataset['e0']
    test_price = df_price[:N_test]
    test_dr = 100*df_y[:N_test]
    train_price = df_price[N_test:(N_test+N_train)]
    train_dr = 100*df_y[N_test:(N_test+N_train)]

    history = model.fit(train_price,train_dr, validation_data = (test_price, test_dr),  epochs=2000, batch_size=32, verbose=0) #validation_data = (test_price, test_dr),
    L = history.history['loss']
    val_L = history.history['val_loss']
    print(f'train loss := {L[-1]}, val loss := {val_L[-1]}')
    np.savez(f'result/baseline/{opts.model}_loss_100', loss=L, val_loss=val_L)
    model.save(f"result/baseline/model_{opts.model}_100")