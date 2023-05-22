import numpy as np
import pandas as pd
import time
import argparse
import cvxpy as cp
import gurobipy
from gurobipy import quicksum
from tqdm import tqdm



def data_generator(
    upperbound_p,
    lowerbound_p,
    upperbound_e,
    lowerbound_e,
    initial_e,
    c1,
    c2,
    efficiency,
    price_hist,
    N=1,
    T=288,
):
    """Generate data from the following optimization problem, some of the args could be writing in kwargs

    Args:
        c1_value (float): linear discharge cost
        c2_value (float): quadratic discharge cost
        upperbound_p (float): power upper bound
        lowerbound_p (float): power lower bound
        upperbound_e (float): energy upper bound
        lowerbound_e (float): energy lower bound
        initial_e (float): inital energy level
        efficiency (float): energy storage efficiency
        price_hist (Dataframe): real-time price
        N (int, optional):  Defaults to 1.
        T (int, optional): Defaults to 288.

    Returns:
        [dataframes]: dataframes of price, discharge and charge
    """
    #

    df_price = np.zeros((N, T))
    df_d = np.zeros((N, T))
    df_p = np.zeros((N, T))
    df_e = np.zeros((N, T+1))
    index = 0
    for ii in tqdm(range(N+100)):
        price = np.array(price_hist.RTP[ii * T*12 : (ii + 1) * T*12])
        if len(price) != T*12:
            continue
        price = np.mean(price.reshape(T,-1),axis=1)
        if min(price) < 0:
            continue
        # price = np.array(price_hist.DAP[i * 288 : (i + 1) * 288 : 12])

        P1 = upperbound_p
        P2 = lowerbound_p
        E1 = upperbound_e
        E2 = lowerbound_e
        e0 = initial_e
        eta = efficiency

        # define the variable and objective function
        model = gurobipy.Model()
        p = model.addVars(T, name='p') # charging
        d = model.addVars(T, name='d') # discharging 
        e = model.addVars(T+1, name='e')
        c = model.addVars(T, name='c')
        
        for i in range(T+1):
            model.addConstr(e[i] == e0 + quicksum(p[j]*eta - d[j]/eta for j in range(i)))
        
        model.addConstrs(c[i]==(e[i]-0.5)*(e[i]-0.5) for i in range(T))

        utility_cost = c1 * quicksum(c[i] for i in range(T)) + c2 * quicksum(d[i] for i in range(T))
        f = quicksum(price[i]*(p[i]-d[i]) for i in range(T)) + utility_cost
        # define constraints
        model.addConstrs(p[i] <= P1 for i in range(T))
        model.addConstrs(p[i] >= P2 for i in range(T))
        model.addConstrs(d[i] <= P1 for i in range(T))
        model.addConstrs(d[i] >= P2 for i in range(T))
        model.addConstrs(e[i] <= E1 for i in range(1,T+1))
        model.addConstrs(e[i] >= E2 for i in range(1,T+1))

        model.setObjective(f, gurobipy.GRB.MINIMIZE)
        model.setParam('OutputFlag', 0)
        model.setParam("NonConvex", 2)
        model.setParam('TimeLimit', 600)
        model.optimize()

        dvalue = np.zeros(T)
        pvalue = np.zeros(T)
        evalue = np.zeros(T+1)
        for k, v in model.getAttr('x', p).items():
            pvalue[k] = v
        for k, v in model.getAttr('x', d).items():
            dvalue[k] = v
        for k, v in model.getAttr('x', e).items():
            evalue[k] = v
        df_price[index, :] = price.T
        df_d[index, :] = dvalue[:]
        df_p[index, :] = pvalue[:]
        df_e[index, :] = evalue[:]
        index = index + 1
        del model
        if index >= N:
            break
    return df_price, df_d, df_p, df_e

if __name__ == '__main__':
    N_train = 240
    dim = 6
    for seed in range(1,10):
        np.random.seed(seed)
        ## initialize parameter
        duration = round(np.random.uniform(1, 4))
        eta = round(np.random.uniform(0.8, 1), 2)
        c1 = round(np.random.uniform(0, 10), 2)
        c2 = round(np.random.uniform(0, 20), 2)
        paras = pd.DataFrame([[0.5, 0.5, c1, c2, duration, eta]],columns=("P1", "E1", "c1", "c2", "duration", "eta"))

        print( "Generating data....",
            "P1=", 0.5,
            "E1=", 0.5 * duration,
            "eta =", eta,
            "c", c1)

        ## load price data
        price_hist = pd.read_csv("../EnergyStorage/dataset/price.csv")
        ## generate dispatch data and save price, true parameters
        df_price, df_d, df_p, df_e = data_generator(
            upperbound_p=0.5,
            lowerbound_p=0,
            upperbound_e=0.5*duration,
            lowerbound_e=0,
            initial_e=0.25*duration,
            c1=c1,
            c2=c2,
            efficiency=eta,
            price_hist=price_hist,
            N=N_train,
            T=dim,
        )
        present_time = time.strftime("%m%d%H%M", time.localtime()) 
        np.savez(f"dataset/version2/data_N{N_train}_{seed}", paras = paras, price = df_price, d=df_d, p=df_p, e=df_e)
