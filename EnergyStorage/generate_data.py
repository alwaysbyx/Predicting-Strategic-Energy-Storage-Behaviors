import numpy as np
import pandas as pd
from utils import data_generator
import time
import argparse




if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--N", type=int, default=150, help="sum of training and valiadation set")


    # opts = parser.parse_args()
    # N_train = opts.N # sum of training and valiadation set
    N_train = 365
    dim = 24
    for seed in range(10):
        np.random.seed(seed)
        ## initialize parameters
        c1_value = round(np.random.uniform(0, 20),2)
        c2_value = round(np.random.uniform(0, 20),2)
        duration = round(np.random.uniform(1, 4))
        eta = round(np.random.uniform(0.8, 1),2)

        paras = pd.DataFrame([[c1_value, c2_value, duration, eta]],columns=("c1", "c2", "P", "eta"))

        print( "Generating data....",
            "P1=", 0.5,
            "E1=", 0.5 * duration,
            "c1 =", c1_value,
            "c2 =", c2_value,
            "eta =", eta,)

        ## load price data
        price_hist = pd.read_csv("dataset/price.csv")

        ## generate dispatch data and save price, true parameters
        df_price, df_d, df_p = data_generator(
            c1_value,
            c2_value,
            upperbound_p=0.5,
            lowerbound_p=0,
            upperbound_e=0.5*duration,
            lowerbound_e=0,
            initial_e=0.25*duration,
            efficiency=eta,
            price_hist=price_hist,
            N=N_train,
            T=dim,
        )

        present_time = time.strftime("%m%d%H%M", time.localtime()) 
        np.savez(f"dataset/data_N{N_train}_{seed}", paras = paras, price = df_price, d=df_d, p=df_p)

