from ctypes import c_int32
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Function, Variable
from torch.nn import Module
import torch.nn as nn
from torch.nn.parameter import Parameter
from util import DRagent_Quad
import time
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)


def train(dataset, N_train=20):
    torch.manual_seed(0)
    df_price = dataset["price"]

    T = 24
    N_train = N_train
    P1 = dataset["p"].max()
    P2 = dataset["d"].max()
    d = dataset["d"]
    p = dataset["p"]
    price_tensor = torch.from_numpy(df_price[10:10+N_train]).double()
    d_tensor = torch.from_numpy(d[10:10+N_train]).double()
    p_tensor = torch.from_numpy(p[10:10+N_train]).double()
    y_tensor = tuple([d_tensor, p_tensor])

    price_tensor2 = torch.from_numpy(df_price[10+N_train:10+int(2*N_train)]).double()
    d_tensor2 = torch.from_numpy(d[10+N_train:10+int(2*N_train)]).double()
    p_tensor2 = torch.from_numpy(p[10+N_train:10+int(2*N_train)]).double()
    y_tensor2 = tuple([d_tensor2, p_tensor2])

    # price_tensor2 = torch.from_numpy(df_price[:10]).double()
    # d_tensor2 = torch.from_numpy(d[:10]).double()
    # p_tensor2 = torch.from_numpy(p[:10]).double()
    # y_tensor2 = tuple([d_tensor2, p_tensor2])
    print('true paras', dataset['paras'])
    cnt = 0
    L = []
    val_L = []
    layer = DRagent_Quad(P1, P2, T)
    opt1 = optim.Adam(layer.parameters(), lr=1e-2)
    pre = time.time()
    for ite in range(200):
        dp_pred = layer(price_tensor)
        if ite == 50:
            opt1.param_groups[0]["lr"] = 1e-2
        elif ite == 200:
            opt1.param_groups[0]["lr"] = 5e-3
        loss = nn.MSELoss()(y_tensor[0], dp_pred[0]) + nn.MSELoss()(y_tensor[1], dp_pred[1])
        opt1.zero_grad()
        loss.backward()
        opt1.step()
        with torch.no_grad():
            layer.E1.data = torch.clamp(layer.E1.data, min=0.01, max=100) 
            layer.E2.data = torch.clamp(layer.E2.data, min=-100, max=-0.01) 
            layer.eta.data =  torch.clamp(layer.eta.data, min=0.8, max=1) 
        layer.eval()
        dp_pred2 = layer(price_tensor2)
        loss2 = nn.MSELoss()(y_tensor2[0], dp_pred2[0]) + nn.MSELoss()(y_tensor2[1], dp_pred2[1])
        if len(val_L) > 1 and loss2.item() > val_L[-1]:
            cnt += 1
        else: cnt = 0
        if cnt >= 30: break
        val_L.append(loss2.detach().numpy())
        print(f'ite = {ite}, loss = {np.round(loss.detach().numpy(),6)}, test_loss = {np.round(val_L[-1],6)}, c1 = {layer.c1.data}, c2={layer.c2.data}, eta = {layer.eta.data}')
        L.append(loss.detach().numpy())
    cur = time.time()
    print(f'Time use: {cur - pre}s')
    L = [float(l) for l in L]
    val_L = [float(l) for l in val_L]
    return L, val_L, layer


if __name__ == "__main__":
    for i in range(1):
        df_dp = np.load(f"dataset/data_N365_{i}.npz")
        L, val_L, layer = train(df_dp, N_train=1)
        torch.save(layer.state_dict(), f'result/model_gradient/quad_model_{i}_N1_noise.pth')
        np.savez(f'result/model_gradient/quad_loss_{i}_N1.npz', loss=L, val_loss=val_L)
        