from ctypes import c_int32
import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
from torch.autograd.functional import hessian
import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter
import util
import plotly.graph_objs as go
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)

class DRagent_Quad(nn.Module):
    """
    Using this layer, we train c1, c2, E1, E2, and eta, other parameters are infered from historical data.
    """
    def __init__(self, P1, P2, T, type='matrix'):
        super().__init__()

        self.E1 = 1.*torch.ones(1) 
        self.E2 = 0.*torch.ones(1) 
        self.eta1 =  nn.Parameter(torch.tensor(0.85)) 
        self.eta2 =  nn.Parameter(torch.tensor(0.95))
        self.T = T
        eps = 1e-4
    
        self.c1 =  nn.Parameter(torch.tensor(1.))
        self.c3 =  nn.Parameter(torch.tensor(1.))
        if type=='matrix':
            self.c2sqrt = nn.Parameter(torch.rand(T,T))
            obj = (lambda d, p, price, c1, c2,  E1, E2, eta1, eta2, e0: -price @ (d - p) + c1 @ d  +  cp.sum_squares(c2 @ d) if isinstance(d, cp.Variable) else -price @ (d - p) + c1 @ d +  torch.sum((c2 @ d)**2))
        else:
            self.c2sqrt = nn.Parameter(torch.tensor(1.))
            self.c4sqrt = nn.Parameter(torch.tensor(1.))
            obj = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: -price @ (d - p) + c1*cp.sum(d) + c3*cp.sum(p) + cp.sum_squares(c2*d) + cp.sum_squares(c4*p)
             if isinstance(d, cp.Variable)
            else -price @ (d - p) +  c1*torch.sum(d) + c3*torch.sum(p) + torch.sum((c2*d)**2) + torch.sum((c4*p)**2))

        self.objective = obj
        self.ineq1 = (lambda d, p, price, c1, c2,  c3, c4,  E1, E2, eta1, eta2, e0: p - torch.ones(T, dtype=torch.double) * P1)
        self.ineq2 = (lambda d, p, price, c1, c2,  c3, c4,  E1, E2, eta1, eta2, e0: torch.ones(T, dtype=torch.double)* 1e-9 - p)
        self.ineq3 = (lambda d, p, price, c1, c2,  c3, c4,  E1, E2, eta1, eta2, e0: d - torch.ones(T, dtype=torch.double) * P2)
        self.ineq4 = (lambda d, p, price, c1, c2,  c3, c4,  E1, E2, eta1, eta2 , e0: torch.ones(T, dtype=torch.double) * 1e-9 - d)
        self.ineq5 = (lambda d, p, price, c1, c2,  c3, c4,  E1, E2, eta1, eta2, e0: torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta1 * p / 4.44 - d / 4.44 / eta2)
            - torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
            - torch.ones(T, dtype=torch.double) * (E1-e0))
        self.ineq6 = lambda d, p, price, c1, c2, c3, c4,   E1, E2, eta1, eta2, e0: torch.ones(
            T, dtype=torch.double
        ) * (E2-e0) - torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta1 * p / 4.4- d /4.4 / eta2) + torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
        
        if type=='matrix':
            parameters = [cp.Parameter(T,), cp.Parameter(T), cp.Parameter((T,T)), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1),]
        else:
            parameters = [cp.Parameter(T),  cp.Parameter(),  cp.Parameter(), cp.Parameter(),  cp.Parameter(), cp.Parameter(1), cp.Parameter(1), cp.Parameter(), cp.Parameter(), cp.Parameter(1),]
        self.layer = util.OptLayer(
            [cp.Variable(T), cp.Variable(T)],
            parameters,
            obj,
            [self.ineq1, self.ineq2, self.ineq3, self.ineq4, self.ineq5, self.ineq6],
            [],
            solver="GUROBI",
            verbose=False,
        )

    def forward(self, price, e0):
        return self.layer(
            price,
            self.c1.expand(price.shape[0], *self.c1.shape),
            self.c2sqrt.expand(price.shape[0], *self.c2sqrt.shape),
            self.c3.expand(price.shape[0], *self.c3.shape),
            self.c4sqrt.expand(price.shape[0], *self.c4sqrt.shape),
            self.E1.expand(price.shape[0], *self.E1.shape),
            self.E2.expand(price.shape[0], *self.E2.shape),
            self.eta1.expand(price.shape[0], *self.eta1.shape),
            self.eta2.expand(price.shape[0], *self.eta2.shape),
            e0
        ) # return: [2, B, T]

def train(dataset, T=24, N_train=20, N_test=10, model_type='matrix'):
    torch.manual_seed(0)
    T = T
    N_train = N_train
    P1 = 1.1
    P2 = 1.1

    df_price = dataset['price']
    df_y = dataset['dr']
    e0 = dataset['e0']
    p = np.clip(df_y, a_min=0, a_max=2)
    d = np.abs(np.clip(df_y, a_min=-2, a_max=0))

    price_tensor = torch.from_numpy(df_price[N_test:N_test+N_train]).double()
    d_tensor = torch.from_numpy(d[N_test:N_test+N_train]).double()
    p_tensor = torch.from_numpy(p[N_test:N_test+N_train]).double()
    y_tensor = tuple([d_tensor, p_tensor])
    e0_tensor = torch.from_numpy(e0[N_test:N_test+N_train]).double()
    #y_tensor = torch.from_numpy(np.mean(df_y[:N_train*288].reshape(N_train, 24, 12),axis=2)).double()

    price_tensor2 = torch.from_numpy(df_price[:N_test]).double()
    d_tensor2 = torch.from_numpy(d[:N_test]).double()
    p_tensor2 = torch.from_numpy(p[:N_test]).double()
    y_tensor2 = tuple([d_tensor2, p_tensor2])
    e0_tensor2 = torch.from_numpy(e0[:N_test]).double()
    #y_tensor2 = torch.from_numpy(np.mean(df_y[N_train*288:(N_train+N_test)*288].reshape(N_test, 24, 12),axis=2)).double()

    L = []
    val_L = []
    layer = DRagent_Quad(P1, P2, T, type=model_type)
    #layer.load_state_dict(torch.load('result/model_gradient/quad/model_scalar1_E.pth'))
    opt1 = optim.Adam(layer.parameters(), lr=1e-1)
    #figure, axes = plt.subplots(1,2, figsize=(8,4))
    for ite in range(200):
        dp_pred = layer(price_tensor, e0_tensor)
        if ite == 50:
            opt1.param_groups[0]["lr"] = 5e-2
        elif ite == 100:
            opt1.param_groups[0]["lr"] = 1e-2
        loss = nn.MSELoss()(100*y_tensor[0], 100*dp_pred[0]) + nn.MSELoss()(100*y_tensor[1], 100*dp_pred[1])
        #loss = nn.MSELoss()(y_tensor, dp_pred[0]-dp_pred[1])

        # anno = plt.annotate(f'step:{ite}', xy=(0.8, 0.9), xycoords='axes fraction',color='black')
        # axes[0].plot(dp_pred[0].detach().numpy()[10]-dp_pred[1].detach().numpy()[10], color='blue')
        # axes[0].plot(d[N_test+10]-p[N_test+10], color='black')
        # axes[1].plot(df_price[N_test+10], color='orange')
        # plt.pause(0.0001)
        # axes[0].clear()
        # anno.remove()

        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            layer.c2sqrt.data = torch.clamp(layer.c2sqrt.data, min=1e-2) 
            layer.c4sqrt.data = torch.clamp(layer.c4sqrt.data, min=1e-2) 
            if ite < 50:
                layer.eta1.data = torch.clamp(layer.eta1.data, min=0.8, max=0.9) 
                layer.eta2.data = torch.clamp(layer.eta2.data, min=0.9, max=1.0) 
                layer.E1.data = torch.clamp(layer.E1.data, min=0.9, max=2.9) 
                layer.E2.data = torch.clamp(layer.E2.data, min=-0.1, max=0.2) 
            else:
                layer.eta1.data = torch.clamp(layer.eta1.data, max=1.0) 
                layer.eta2.data = torch.clamp(layer.eta2.data, max=1.0) 
        layer.eval()
        dp_pred2 = layer(price_tensor2, e0_tensor2)
        loss2 = nn.MSELoss()(100*y_tensor2[0], 100*dp_pred2[0]) + nn.MSELoss()(100*y_tensor2[1], 100*dp_pred2[1])
        #loss2 = nn.MSELoss()(y_tensor2, dp_pred2[0]-dp_pred2[0])
        val_L.append(loss2.detach().numpy())
        print(f'ite = {ite}, loss = {loss.detach().numpy()}, val_l = {loss2.detach().numpy()}, eta1 = {layer.eta1.detach().numpy()}, eta2 = {layer.eta2.detach().numpy()}, E1 = {layer.E1.data}, E2 = {layer.E2.data}')
        L.append(loss.detach().numpy())
    
    L = [float(l) for l in L]
    val_L = [float(l) for l in val_L]
    return L, val_L, layer


if __name__ == '__main__':
    model_type = 'scalar'
    #data = pd.read_csv('dataset/UQ Tesla Battery - 2022 (5-min resolution).csv')  
    data = np.load(f"dataset/UQ-Tesla-2020.npz")
    L, val_L, layer = train(data, T=80, N_train=22, N_test=20, model_type=model_type)
    torch.save(layer.state_dict(), f'result/model_gradient/quad/model_{model_type}1_E.pth')
    np.savez(f'result/model_gradient/quad/loss_{model_type}1_E.npz', loss=L, val_loss=val_L)