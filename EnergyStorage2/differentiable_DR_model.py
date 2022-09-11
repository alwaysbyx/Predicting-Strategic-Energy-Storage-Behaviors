from ctypes import c_int32
import torch
import torch.nn as nn
from functorch import vmap
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from itertools import accumulate
import cvxpy as cp
import gurobipy
import numpy as np
from torch.autograd.functional import hessian
import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter
import util
from collections import OrderedDict
import plotly.graph_objs as go
import random
import pandas as pd
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)


class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""
    def __init__(self, hidden_num=24):
        super().__init__()

        self.linear_z1 = nn.Linear(hidden_num, hidden_num, bias=False)
        self.linear_y0 = nn.Linear(1, hidden_num)
        self.linear_y1 = nn.Linear(1, 1)
        torch.nn.init.uniform_(self.linear_z1.weight, a=1e-2,b=4)

        self.act = nn.Softplus()

    def forward(self, y):
        z1 = self.linear_y0(y)
        z = self.act(self.linear_z1(z1) + self.linear_y1(y))
        return torch.sum(z,axis=1)

class DRagent(nn.Module):
    """
    Using this layer, we train c1, c2, E1, E2, and eta, other parameters are infered from historical data.
    """
    def __init__(self, P1, P2, T):
        super().__init__()

        self.E1 = nn.Parameter(0.5 * torch.ones(1))
        self.E2 = nn.Parameter(-0.5 * torch.ones(1))
        self.eta = nn.Parameter(0.9 * torch.ones(1))
        self.T = T
        eps = 1e-4

        obj = (
            lambda d, p, price, c1, c2, E1, E2, eta: -price @ (d - p)
            + c1 @ d
            + cp.sum_squares(cp.sqrt(cp.diag(c2)) @ d)
            if isinstance(d, cp.Variable)
            else -price @ (d - p) + c1 @ d + torch.sum((torch.sqrt(torch.diag(c2)) @ d)**2)
        )
        self.objective = obj
        self.costNN = ICNN()

        self.ineq1 = (lambda d, p, price, c1, c2, E1, E2, eta: p - torch.ones(T, dtype=torch.double) * P1)
        self.ineq2 = (lambda d, p, price, c1, c2, E1, E2, eta: torch.ones(T, dtype=torch.double)* 1e-9 - p)
        self.ineq3 = (lambda d, p, price, c1, c2, E1, E2, eta: d - torch.ones(T, dtype=torch.double) * P2)
        self.ineq4 = (lambda d, p, price, c1, c2, E1, E2, eta: torch.ones(T, dtype=torch.double) * 1e-9 - d)
        self.ineq5 = (lambda d, p, price, c1, c2, E1, E2, eta: torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta * p - d / eta)
            - torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
            - torch.ones(T, dtype=torch.double) * E1)
        self.ineq6 = lambda d, p, price, c1, c2, E1, E2, eta: torch.ones(
            T, dtype=torch.double
        ) * E2 - torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta * p - d / eta) + torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
        

        self.layer = util.OptLayer(
            [cp.Variable(T), cp.Variable(T)],
            [
                cp.Parameter(T,),
                cp.Parameter(T),
                cp.Parameter(T),
                cp.Parameter(1),
                cp.Parameter(1),
                cp.Parameter(1),
            ],
            obj,
            [self.ineq1, self.ineq2, self.ineq3, self.ineq4, self.ineq5, self.ineq6],
            [],
            solver="GUROBI",
            verbose=False,
        )
    
    def approximate_cost(self, y, diff=True):
        # y: demand response [T,B,1]
        T = self.T
        Cf = self.costNN
        with torch.enable_grad():
            tau = y.data
            tau = Variable(tau, requires_grad=True)
            costs = list()
            hessians = list()
            grads = list()
            for t in range(T):
                tau_t = tau[t]
                cost = Cf(tau_t)
                grad = torch.autograd.grad(cost.sum(), tau_t, create_graph=True, retain_graph=True)[0]
                hessian = list()
                for v_i in range(tau.shape[2]):
                    hessian.append(
                        torch.autograd.grad(grad[:, v_i].sum(), tau_t,
                                            retain_graph=True)[0]
                    )
                hessian = torch.stack(hessian, dim=-1)
                costs.append(cost)
                grads.append(grad - util.bmv(hessian, tau_t))
                hessians.append(hessian)
            costs = torch.stack(costs, dim=0)
            grads = torch.stack(grads, dim=0)
            hessians = torch.stack(hessians, dim=0)
            hessians = hessians.squeeze().permute(1,0) # [B,T]
            grads = grads.squeeze().permute(1,0) # [B,T]
            if not diff:
                return hessians.data, grads.data, costs.data
            return hessians, grads, costs


    def forward(self, price, dr, ite):
        # price: [B,T]
        # dr: [B, T]
        if ite < 500:
            d = torch.rand(price.shape[1], price.shape[0]).unsqueeze(2) # [T,B,1]
        else:
            d = dr.permute(1,0).unsqueeze(2)
        for i in range(10):
            c2, c1, _ = self.approximate_cost(d, diff=False)
            if torch.min(c2) <= 1e-5:
                c2.data = torch.clamp(c2.data, min=1e-5) 
            d, p = self.layer(price, c1, c2,
                            self.E1.expand(price.shape[0], *self.E1.shape),
                            self.E2.expand(price.shape[0], *self.E2.shape),
                            self.eta.expand(price.shape[0], *self.eta.shape), flag=True)
            d = d.permute(1,0).unsqueeze(2)
        
        c2, c1, _ = self.approximate_cost(d, diff=True) # c2: [B,T], c1: [B,T]

        return self.layer(
            price,
            c1, 
            c2,
            self.E1.expand(price.shape[0], *self.E1.shape),
            self.E2.expand(price.shape[0], *self.E2.shape),
            self.eta.expand(price.shape[0], *self.eta.shape),
        ) # return: [2, B, T]



def train(dataset, N_train=20):
    df_price = dataset["price"]

    T = 24
    N_train = 20
    P1 = dataset["p"].max()
    P2 = dataset["d"].max()
    d = dataset["d"]
    p = dataset["p"]
    price_tensor = torch.from_numpy(df_price[10:10+N_train]).double()
    d_tensor = torch.from_numpy(d[10:10+N_train]).double()
    p_tensor = torch.from_numpy(p[10:10+N_train]).double()
    y_tensor = tuple([d_tensor, p_tensor])

    price_tensor2 = torch.from_numpy(df_price[:10]).double()
    d_tensor2 = torch.from_numpy(d[:10]).double()
    p_tensor2 = torch.from_numpy(p[:10]).double()
    y_tensor2 = tuple([d_tensor2, p_tensor2])

    L = []
    val_L = []
    layer = DRagent(P1, P2, T)
    opt1 = optim.Adam(layer.parameters(), lr=2e-2)
    for ite in range(500):
        dp_pred = layer(price_tensor, d_tensor, ite)
        if ite == 5:
            opt1.param_groups[0]["lr"] = 5e-3
        loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            layer.E1.data = torch.clamp(layer.E1.data, min=0.01, max=100) 
            layer.E2.data = torch.clamp(layer.E2.data, min=-100, max=-0.01) 
            layer.eta.data =  torch.clamp(layer.eta.data, min=0.5, max=1) 
        layer.eval()
        dp_pred2 = layer(price_tensor2, d_tensor2, 500)
        loss2 = nn.MSELoss()(y_tensor2[0], dp_pred2[0]) + nn.MSELoss()(y_tensor2[1], dp_pred2[1])
        val_L.append(loss2.detach().numpy())
        print(f'ite = {ite}, loss = {loss.detach().numpy()}, eta = {layer.eta.data}')
        L.append(loss.detach().numpy())
    
    L = [float(l) for l in L]
    val_L = [float(l) for l in val_L]
    return L, val_L, layer


if __name__ == '__main__':
    for i in range(7,10):
        if i == 1: continue
        df_dp = np.load(f"dataset/data_N365_{i}.npz")
        L, val_L, layer = train(df_dp)
        torch.save(layer.state_dict(), f'result/model_gradient/ICNN/model_{i}.pth')
        np.savez(f'result/model_gradient/ICNN/loss_{i}.npz', loss=L, val_loss=val_L)