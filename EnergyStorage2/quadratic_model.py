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
        self.eta = nn.Parameter(0.9 * torch.ones(1))  #eta*torch.ones(1) #nn.Parameter(0.9 * torch.ones(1))
        self.T = T
        eps = 1e-4
    
        self.c1 = nn.Parameter(torch.rand(T))
        if type=='matrix':
            self.c2sqrt = nn.Parameter(torch.rand(T,T))
            obj = (lambda d, p, price, c1, c2, E1, E2, eta: -price @ (d - p) + c1 @ d  +  cp.sum_squares(c2 @ d) if isinstance(d, cp.Variable) else -price @ (d - p) + c1 @ d +  torch.sum((c2 @ d)**2) )
        else:
            self.c2sqrt = nn.Parameter(torch.ones(T))
            obj = (lambda d, p, price, c1, c2, E1, E2, eta: -price @ (d - p) + c1 @ d  + cp.sum_squares(cp.sqrt(cp.diag(c2)) @ d) if isinstance(d, cp.Variable)
            else -price @ (d - p) + c1 @ d + torch.sum((torch.sqrt(torch.diag(c2)) @ d)**2))

        self.objective = obj
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
        
        if type=='matrix':
            parameters = [cp.Parameter(T,), cp.Parameter(T), cp.Parameter((T,T)), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1),]
        else:
            parameters = [cp.Parameter(T,), cp.Parameter(T), cp.Parameter(T), cp.Parameter(1), cp.Parameter(1), cp.Parameter(1),]
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
    df_price = dataset["price"]
    eta = dataset['paras'][0][-1]
    print('*'*10, 'eta=',eta, dataset['paras'])
    T = T
    N_train = N_train
    P1 = dataset["p"].max()
    P2 = dataset["d"].max()
    d = dataset["d"]
    p = dataset["p"]
    price_tensor = torch.from_numpy(df_price[N_test:N_test+N_train]).double()
    d_tensor = torch.from_numpy(d[N_test:N_test+N_train]).double()
    p_tensor = torch.from_numpy(p[N_test:N_test+N_train]).double()
    y_tensor = tuple([d_tensor, p_tensor])

    price_tensor2 = torch.from_numpy(df_price[:N_test]).double()
    d_tensor2 = torch.from_numpy(d[:N_test]).double()
    p_tensor2 = torch.from_numpy(p[:N_test]).double()
    y_tensor2 = tuple([d_tensor2, p_tensor2])

    L = []
    val_L = []
    layer = DRagent_Quad(P1, P2, T, eta, type=model_type)
    opt1 = optim.Adam(layer.parameters(), lr=2e-2)
    for ite in range(500):
        dp_pred = layer(price_tensor)
        if ite == 10:
            opt1.param_groups[0]["lr"] = 5e-3
        elif ite == 150:
            opt1.param_groups[0]["lr"] = 5e-3
        loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            layer.E1.data = torch.clamp(layer.E1.data, min=0.01, max=100) 
            layer.E2.data = torch.clamp(layer.E2.data, min=-100, max=-0.01) 
            layer.eta.data =  torch.clamp(layer.eta.data, min=0.5, max=1) 
            layer.c2sqrt.data = torch.clamp(layer.c2sqrt.data, min=1e-2) 
        layer.eval()
        dp_pred2 = layer(price_tensor2)
        loss2 = nn.MSELoss()(y_tensor2[0], dp_pred2[0]) + nn.MSELoss()(y_tensor2[1], dp_pred2[1])
        val_L.append(loss2.detach().numpy())
        print(f'ite = {ite}, loss = {loss.detach().numpy()}, eta = {layer.eta.data}, val_l = {loss2.detach().numpy()}')
        L.append(loss.detach().numpy())
    
    L = [float(l) for l in L]
    val_L = [float(l) for l in val_L]
    return L, val_L, layer


if __name__ == '__main__':
    model_type = 'matrix'
    for i in range(1,10):
        #df_dp = np.load(f"dataset/version1/data_N365_5.npz")
        df_dp = np.load(f"dataset/version2/data_N240_{i}.npz")
        L, val_L, layer = train(df_dp, T=6, N_train=40, N_test=10, model_type=model_type)
        torch.save(layer.state_dict(), f'result/model_gradient/quad/version2/model_{i}.pth')
        np.savez(f'result/model_gradient/quad/version2/loss_{i}.npz', loss=L, val_loss=val_L)