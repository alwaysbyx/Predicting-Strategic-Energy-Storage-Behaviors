from ctypes import c_int32
from pyexpat import model
from turtle import hideturtle
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
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)


class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""
    def __init__(self, input_num=48, hidden_num=24):
        super().__init__()

        self.linear_z1 = nn.Linear(hidden_num, 1, bias=False)
        self.linear_y0 = nn.Linear(input_num, hidden_num)
        self.linear_y1 = nn.Linear(input_num, 1)
        torch.nn.init.uniform_(self.linear_z1.weight, a=1e-2,b=4)
        torch.nn.init.uniform_(self.linear_y0.weight, a=-2,b=2)
        self.act = nn.Softplus()

    def forward(self, y):
        z1 = self.act(self.linear_y0(y))
        z2 = self.act(self.linear_z1(z1) + self.linear_y1(y))
        return z2



class ICNN2(torch.nn.Module): # based on scalar
    """Input Convex Neural Network"""
    def __init__(self, hidden_num=6):
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
    def __init__(self, P1, P2, T, type='scalar'):
        super().__init__()

        self.E1 = 1.*torch.ones(1) # nn.Parameter(0.5 * torch.ones(1))
        self.E2 = 0.*torch.ones(1) # nn.Parameter(-0.5 * torch.ones(1))
        self.eta1 = nn.Parameter(0.95 * torch.ones(1)) # 0.85*torch.ones(1) #eta*torch.ones(1) #nn.Parameter(0.9 * torch.ones(1))
        self.eta2 =  nn.Parameter(0.85 * torch.ones(1)) # 0.95*torch.ones(1)
        self.T = T
        eps = 1e-4
        
        self.type = type
        if type=='scalar':
            self.costNN = ICNN(input_num=1)
            obj = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: -price @ (d - p) + c1 @ d + cp.sum_squares(cp.sqrt(cp.diag(c2)) @ d) +cp.sum_squares(cp.sqrt(cp.diag(c4)) @ p) + c3@p
            if isinstance(d, cp.Variable)
            else -price @ (d - p) + c1 @ d +  torch.sum((torch.sqrt(torch.diag(c2)) @ d)**2) + +c3 @ p + torch.sum((torch.sqrt(torch.diag(c4)) @ p)**2))
        else: 
            self.costNN = ICNN(input_num=T)
            obj = (lambda d, p, price, c1, c2, c3, c4,  E1, E2, eta1, eta2, e0: -price @ (d - p) + c1 @ d  +  cp.QuadForm(d, c2) + c3 @ p  +  cp.QuadForm(p, c4) if isinstance(d, cp.Variable) else -price @ (d - p) + c1 @ d + c3 @ p +  d @ c2 @ d + p @ c4 @ p  )
        self.objective = obj
        self.ineq1 = (lambda d, p, price, c1, c2, c3, c4,  E1, E2, eta1, eta2, e0: p - torch.ones(T, dtype=torch.double) * P1)
        self.ineq2 = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: torch.ones(T, dtype=torch.double) * 1e-9 - p)
        self.ineq3 = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: d-torch.ones(T, dtype=torch.double) * P2)
        self.ineq4 = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: torch.ones(T, dtype=torch.double) * 1e-9 - d)
        self.ineq5 = (lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta1 /4.44 * p - d /4.44/ eta2)
            - torch.as_tensor(np.arange(eps, (T+1)*eps, eps))
            - torch.ones(T, dtype=torch.double) * (E1-e0))
        self.ineq6 = lambda d, p, price, c1, c2, c3, c4, E1, E2, eta1, eta2, e0: torch.ones(
            T, dtype=torch.double
        ) * (E2-e0) - torch.tril(torch.ones(T, T, dtype=torch.double)) @ (eta1 * p / 4.44 - d / eta2 / 4.44) + torch.as_tensor(np.arange(eps, (T+1)*eps, eps))

        if self.type == 'scalar':
            parameters = [cp.Parameter(T,), cp.Parameter(T),  cp.Parameter(T), cp.Parameter(T),cp.Parameter(T), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1)]
        else: 
            parameters = [cp.Parameter(T,), cp.Parameter(T), cp.Parameter((T,T),PSD=True),  cp.Parameter(T), cp.Parameter((T,T),PSD=True), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1)]
        self.layer = util.OptLayer(
            [cp.Variable(T), cp.Variable(T)],
            parameters,
            obj,
            [self.ineq1, self.ineq2, self.ineq3, self.ineq4, self.ineq5, self.ineq6],
            [],
            solver="GUROBI",
            verbose=False,
        )

    def approximate_cost(self, y,  diff=True):
        # y: demand response [T,B,1]
        Cf = self.costNN
        with torch.enable_grad():
            tau = y.data
            #tau2 = p.data
            #tau = torch.concat([tau,tau2],dim=1)
            tau = Variable(tau, requires_grad=True)
            cost = Cf(tau)
            grad = torch.autograd.grad(cost.sum(), tau, create_graph=True, retain_graph=True)[0] # [B,T]
            hessian = list()
            for i in range(self.T):
                hessian.append(torch.autograd.grad(grad[:,i].sum(), tau, retain_graph=True)[0])
            hessian = torch.stack(hessian, dim=-1) # [B,T,T]
            if not diff:
                return hessian.data, grad.data
            return hessian, grad
    
    
    def approximate_cost2(self, y, diff=True): # based_on_scalar
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


    def forward(self, price, e0, trued, truep):
        # price: [B,T]
        # dr: [B, T]
        #figure, axes = plt.subplots(1,2, figsize=(8,4))
        if self.type == 'scalar':
            d = trued.permute(1,0).unsqueeze(2) #torch.rand(price.shape[1], price.shape[0]).unsqueeze(2) #trued.permute(1,0).unsqueeze(2)
            p = truep.permute(1,0).unsqueeze(2) #torch.rand(price.shape[1], price.shape[0]).unsqueeze(2) #truep.permute(1,0).unsqueeze(2)
        else:
            d = trued
            p = truep
        for i in range(15):
            if self.type == 'scalar':
                c2, c1, _ = self.approximate_cost2(d, diff=False)
                c4, c3, _ = self.approximate_cost2(p, diff=False)
            else:
                c2, c1 = self.approximate_cost(d, diff=False)
                c4, c3 = self.approximate_cost(p, diff=False)
            d, p = self.layer(price, c1, c2, c3, c4,
                            self.E1.expand(price.shape[0], *self.E1.shape),
                            self.E2.expand(price.shape[0], *self.E2.shape),
                            self.eta1.expand(price.shape[0], *self.eta1.shape), 
                            self.eta2.expand(price.shape[0], *self.eta2.shape),
                            e0, flag=True)
            # anno = plt.annotate(f'step:{i}', xy=(0.8, 0.9), xycoords='axes fraction',color='black')
            # axes[0].plot(d.detach().numpy()[2], color='blue')
            # axes[0].plot(p.detach().numpy()[2], color='black')
            # plt.pause(0.0001)
            # axes[0].clear()
            # anno.remove()
            if self.type == 'scalar':
                d = d.permute(1,0).unsqueeze(2)
                p = p.permute(1,0).unsqueeze(2)
        if self.type == 'scalar':
            c2, c1, _ = self.approximate_cost2(d, diff=True)
            c4, c3, _ = self.approximate_cost2(p, diff=True)
        else:
            c2, c1 = self.approximate_cost(d, diff=True)
            c4, c3 = self.approximate_cost(p, diff=True)
        return self.layer(
            price, c1, c2, c3, c4,
            self.E1.expand(price.shape[0], *self.E1.shape),
            self.E2.expand(price.shape[0], *self.E2.shape),
            self.eta1.expand(price.shape[0], *self.eta1.shape), 
            self.eta2.expand(price.shape[0], *self.eta2.shape),
            e0, 
        ) # return: [2, B, T]

        #axes[0].plot(d.detach().numpy()[2], color='blue')
        #axes[0].plot(p.detach().numpy()[2], color='black')
        plt.pause(0)

        return d, p


def train(dataset, i, T=24,  N_train=20, N_test=10,  model_type='vector'):
    torch.manual_seed(i)
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

    L = []
    val_L = []
    layer = DRagent(P1, P2, T, type=model_type)
    opt1 = optim.Adam(layer.parameters(), lr=1e-1)
    #figure, axes = plt.subplots(1,2, figsize=(8,4))
    for ite in range(200):
        dp_pred = layer(price_tensor, e0_tensor, d_tensor, p_tensor)
        if ite == 10:
            opt1.param_groups[0]["lr"] = 5e-2
        elif ite == 50:
            opt1.param_groups[0]["lr"] = 1e-2
        elif ite == 100:
            opt1.param_groups[0]["lr"] = 5e-3
        loss = nn.MSELoss()(100*y_tensor[0], 100*dp_pred[0]) + nn.MSELoss()(100*y_tensor[1], 100*dp_pred[1])
        #loss = nn.MSELoss()(y_tensor, dp_pred[0]-dp_pred[1])
        # axes[0].plot(y_tensor[0].detach().numpy()[0]-y_tensor[1].detach().numpy()[0], color='orange')
        # axes[0].plot(dp_pred[0].detach().numpy()[2], color='blue')
        # axes[0].plot(dp_pred[1].detach().numpy()[2], color='black')
        # plt.pause(0.0001)
        # axes[0].clear()
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            if torch.min(layer.costNN.linear_z1.weight.data) < 1e-3:
                layer.costNN.linear_z1.weight.data = torch.clamp(layer.costNN.linear_z1.weight.data, min=1e-2)
            if ite < 50:
                layer.eta1.data = torch.clamp(layer.eta1.data, min=0.84,max=0.86)
                layer.eta2.data = torch.clamp(layer.eta2.data, min=0.94,max=0.96)
        layer.eval()
        dp_pred2 = layer(price_tensor2, e0_tensor2, d_tensor, p_tensor)
        loss2 = nn.MSELoss()(100*y_tensor2[0], 100*dp_pred2[0]) + nn.MSELoss()(100*y_tensor2[1], 100*dp_pred2[1])
        #loss2 = nn.MSELoss()(y_tensor2, dp_pred2[0]-dp_pred2[0])
        val_L.append(loss2.detach().numpy())
        print(f'ite = {ite}, loss = {loss.detach().numpy()}, eta1={layer.eta1.detach().numpy()}, eta2={layer.eta2.detach().numpy()}, val_l = {loss2.detach().numpy()}')
        L.append(loss.detach().numpy())
        if ite % 50 == 0:
            torch.save(layer.state_dict(), f'result/model_gradient/ICNN/model_{model_type}.pth')
    
    L = [float(l) for l in L]
    val_L = [float(l) for l in val_L]
    return L, val_L, layer

if __name__ == '__main__':
    model_type = 'vector'
    #data = pd.read_csv('dataset/UQ Tesla Battery - 2022 (5-min resolution).csv')  
    #L, val_L, layer = train(data, T=24, N_train=40, N_test=10, model_type=model_type)
    data = np.load(f"dataset/UQ-Tesla-2020.npz")
    L, val_L, layer = train(data, 0, T=80, N_train=22, N_test=20, model_type=model_type)
    torch.save(layer.state_dict(), f'result/model_gradient/ICNN/model_{model_type}.pth')
    np.savez(f'result/model_gradient/ICNN/loss_{model_type}a.npz', loss=L, val_loss=val_L)