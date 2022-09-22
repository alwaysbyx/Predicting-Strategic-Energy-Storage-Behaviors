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
import pandas as pd
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)

class DRagent_Quad(nn.Module):
    """
    Using this layer, we train c1, c2, E1, E2, and eta, other parameters are infered from historical data.
    """
    def __init__(self, P1, P2, T, eta, type='matrix'):
        super().__init__()

        self.E1 = nn.Parameter(0.5 * torch.ones(1))
        self.E2 = nn.Parameter(-0.5 * torch.ones(1))
        self.eta = eta*torch.ones(1) #eta*torch.ones(1) #nn.Parameter(0.9 * torch.ones(1))
        self.T = T
        eps = 1e-4
    
        self.c1 = nn.Parameter(torch.rand(T))
        self.c3 = nn.Parameter(torch.rand(T))
        if type=='matrix':
            self.c2sqrt = nn.Parameter(torch.rand(T,T))
            obj = (lambda d, p, price, c1, c2,  E1, E2, eta: -price @ (d - p) + c1 @ d  +  cp.sum_squares(c2 @ d) + 5 @ ((torch.ones(T, dtype=torch.double) * P1)-d) if isinstance(d, cp.Variable) else -price @ (d - p) + c1 @ d +  torch.sum((c2 @ d)**2) + 5 @ ((torch.ones(T, dtype=torch.double) * P1)-d))
        else:
            self.c2sqrt = nn.Parameter(torch.ones(T))
            self.c4sqrt = nn.Parameter(torch.ones(T))
            obj = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta: -price @ (d - p) + c1 @ d + c3 @ p + cp.sum_squares(cp.sqrt(cp.diag(c2)) @ d) + cp.sum_squares(cp.sqrt(cp.diag(c4)) @ p)
             if isinstance(d, cp.Variable)
            else -price @ (d - p) + c1 @ d + c3 @ p + torch.sum((torch.sqrt(torch.diag(c2)) @ d)**2) + torch.sum((torch.sqrt(torch.diag(c4)) @ p)**2))

        self.objective = obj
        self.ineq1 = (lambda d, p, price, c1, c2,  E1, E2, eta: p - torch.ones(T, dtype=torch.double) * P1)
        self.ineq2 = (lambda d, p, price, c1, c2,  E1, E2, eta: torch.ones(T, dtype=torch.double)* 1e-9 - p)
        self.ineq3 = (lambda d, p, price, c1, c2,  E1, E2, eta: d - torch.ones(T, dtype=torch.double) * P2)
        self.ineq4 = (lambda d, p, price, c1, c2,  E1, E2, eta: torch.ones(T, dtype=torch.double) * 1e-9 - d)
        self.ineq5 = (lambda d, p, price, c1, c2,  E1, E2, eta: torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta * p - d / eta)
            - torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
            - torch.ones(T, dtype=torch.double) * E1)
        self.ineq6 = lambda d, p, price, c1, c2,  E1, E2, eta: torch.ones(
            T, dtype=torch.double
        ) * E2 - torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta * p - d / eta) + torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
        
        if type=='matrix':
            parameters = [cp.Parameter(T,), cp.Parameter(T), cp.Parameter((T,T)), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1),]
        else:
            parameters = [cp.Parameter(T,), cp.Parameter(T), cp.Parameter(T),  cp.Parameter(T),  cp.Parameter(T), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1),]
        self.layer = util.OptLayer(
            [cp.Variable(T), cp.Variable(T)],
            parameters,
            obj,
            [self.ineq1, self.ineq2, self.ineq3, self.ineq4, self.ineq5, self.ineq6],
            [],
            solver="GUROBI",
            verbose=False,
        )

    def forward(self, price):
        return self.layer(
            price,
            self.c1.expand(price.shape[0], *self.c1.shape),
            self.c2sqrt.expand(price.shape[0], *self.c2sqrt.shape),
            self.E1.expand(price.shape[0], *self.E1.shape),
            self.E2.expand(price.shape[0], *self.E2.shape),
            self.eta.expand(price.shape[0], *self.eta.shape),
        ) # return: [2, B, T]

def train(dataset, T=24, N_train=20, N_test=10, model_type='matrix'):
    torch.manual_seed(0)
    T = T
    N_train = N_train
    P1 = 1.9
    P2 = 1.9

    df_price = dataset['Energy Price ($/MWh)'].values
    df_y = dataset['3 Phase Power (kW)'].values / 1000
    d = np.clip(df_y, a_min=0, a_max=2)
    p = np.abs(np.clip(df_y, a_min=-2, a_max=0))

    price_tensor = torch.from_numpy(np.mean(df_price[:N_train*288].reshape(N_train, 24, 12),axis=2)).double()
    d_tensor = torch.from_numpy(np.mean(d[:N_train*288].reshape(N_train, 24, 12),axis=2)).double()
    p_tensor = torch.from_numpy(np.mean(p[:N_train*288].reshape(N_train, 24, 12),axis=2)).double()
    y_tensor = tuple([d_tensor, p_tensor])
    #y_tensor = torch.from_numpy(np.mean(df_y[:N_train*288].reshape(N_train, 24, 12),axis=2)).double()

    price_tensor2 = torch.from_numpy(np.mean(df_price[N_train*288:(N_train+N_test)*288].reshape(N_test, 24, 12),axis=2)).double()
    d_tensor2 = torch.from_numpy(np.mean(d[N_train*288:(N_train+N_test)*288].reshape(N_test, 24, 12),axis=2)).double()
    p_tensor2 = torch.from_numpy(np.mean(p[N_train*288:(N_train+N_test)*288].reshape(N_test, 24, 12),axis=2)).double()
    y_tensor2 = tuple([d_tensor2, p_tensor2])
    #y_tensor2 = torch.from_numpy(np.mean(df_y[N_train*288:(N_train+N_test)*288].reshape(N_test, 24, 12),axis=2)).double()

    L = []
    val_L = []
    layer = DRagent_Quad(P1, P2, T, 0.865, type=model_type)
    opt1 = optim.Adam(layer.parameters(), lr=2e-2)
    for ite in range(500):
        dp_pred = layer(price_tensor)
        if ite == 50:
            opt1.param_groups[0]["lr"] = 1e-2
        elif ite == 150:
            opt1.param_groups[0]["lr"] = 5e-3
        loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
        #loss = nn.MSELoss()(y_tensor, dp_pred[0]-dp_pred[1])
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            layer.E1.data = torch.clamp(layer.E1.data, min=0.01, max=100) 
            layer.E2.data = torch.clamp(layer.E2.data, min=-100, max=-0.01) 
            layer.c2sqrt.data = torch.clamp(layer.c2sqrt.data, min=1e-2) 
        layer.eval()
        dp_pred2 = layer(price_tensor2)
        loss2 = nn.MSELoss()(y_tensor2[0], dp_pred2[0]) + nn.MSELoss()(y_tensor2[1], dp_pred2[1])
        #loss2 = nn.MSELoss()(y_tensor2, dp_pred2[0]-dp_pred2[0])
        val_L.append(loss2.detach().numpy())
        print(f'ite = {ite}, loss = {loss.detach().numpy()}, eta = {layer.eta.data}, val_l = {loss2.detach().numpy()}')
        L.append(loss.detach().numpy())
    
    L = [float(l) for l in L]
    val_L = [float(l) for l in val_L]
    return L, val_L, layer


if __name__ == '__main__':
    model_type = 'matrix'
    data = pd.read_csv('dataset/UQ Tesla Battery - 2022 (5-min resolution).csv')  
    L, val_L, layer = train(data, T=24, N_train=40, N_test=10, model_type=model_type)
    torch.save(layer.state_dict(), f'result/model_gradient/quad/model_{model_type}.pth')
    np.savez(f'result/model_gradient/quad/loss_{model_type}.npz', loss=L, val_loss=val_L)