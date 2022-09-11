import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
from itertools import accumulate
import cvxpy as cp
import gurobipy
import numpy as np
from functorch import jacrev, grad

import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter
import operator

def bmv(X, y):
    return X.bmm(y.unsqueeze(2)).squeeze(2)


class OptLayer(nn.Module):
    """Identical to the example in deep implicity layer tutorial
    http://implicit-layers-tutorial.org/differentiable_optimization/
    """

    def __init__(
        self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts
    ):
        super().__init__()
        self.variables = variables
        self.parameters = parameters
        self.objective = objective
        self.inequalities = inequalities
        self.equalities = equalities
        self.cvxpy_opts = cvxpy_opts
        # create the cvxpy problem with objective, inequalities, equalities
        self.cp_inequalities = [
            ineq(*variables, *parameters) <= 0 for ineq in inequalities
        ]
        self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
        self.problem = cp.Problem(
            cp.Minimize(objective(*variables, *parameters)),
            self.cp_inequalities + self.cp_equalities,
        )

    def forward(self, *batch_params, flag = False):
        out, J = [], []
        # solve over minibatch by just iterating
        for batch in range(batch_params[0].shape[0]):
            # solve the optimization problem and extract solution + dual variables
            # print(self.cp_inequalities + self.cp_equalities)
            params = [p[batch] for p in batch_params]
            with torch.no_grad():
                for i, p in enumerate(self.parameters):
                    p.value = params[i].double().numpy()
                    # if i == 2: 
                    #     print(np.shape(p.value))
                    #     p.value = (p.value+p.value.T)/2
                    #     print(np.all(p.value==p.value.T))
                    #     print(np.all(np.linalg.eigvals(p.value)>0))
                self.problem.solve(**self.cvxpy_opts)
                z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
                lam = [
                    torch.tensor(c.dual_value).type_as(params[0])
                    for c in self.cp_inequalities
                ]
                nu = [
                    torch.tensor(c.dual_value).type_as(params[0])
                    for c in self.cp_equalities
                ]
                
            # convenience routines to "flatten" and "unflatten" (z,lam,nu)
            def vec(z, lam, nu):
                return torch.cat([a.view(-1) for b in [z, lam, nu] for a in b])
            def mat(x):
                sz = [0] + list(
                    accumulate([a.numel() for b in [z, lam, nu] for a in b])
                )
                val = [x[a:b] for a, b in zip(sz, sz[1:])]
                return (
                    [val[i].view_as(z[i]) for i in range(len(z))],
                    [val[i + len(z)].view_as(lam[i]) for i in range(len(lam))],
                    [val[i + len(z) + len(lam)].view_as(nu[i]) for i in range(len(nu))],
                )

            # computes the KKT residual
            def kkt(z, lam, nu, *params):
                
                g = [ineq(*z, *params) for ineq in self.inequalities]
                dnu = [eq(*z, *params) for eq in self.equalities]
                # print("lagrang. partial w.r. to eq dual: " + str(time.time() - t) + " s")
                L = (
                    self.objective(*z, *params)
                    + sum((u * v).sum() for u, v in zip(lam, g))
                    + sum((u * v).sum() for u, v in zip(nu, dnu))
                )

                dz = autograd.grad(L, z, create_graph=True)
                # print("lagrang. partial w.r. to primal: " + str(time.time() - tz) + " s")
                dlam = [lam[i] * g[i] for i in range(len(lam))]
                # print("lagrang. partial w.r. to dual ineq: " + str(time.time() - tg) + " s")
                
                return dz, dlam, dnu


            y = vec(z, lam, nu)
            if flag:
                out.append(mat(y)[0])
                continue
            y = y - vec(
                *kkt(
                    [z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params
                )
            )

            # compute jacobian and backward hook
            currjac = jacrev(lambda x: vec(*kkt(*mat(x), *params)))(y)
            # print("jacobian (inc. kkt residuals): " + str(time.time() - tjac) + " s")
            # print("time to compute jacobian: " + str(time.time()-t) + "s econds")
            reg = torch.eye(n = currjac.shape[0], m = currjac.shape[1])
            J.append(currjac)
            
            # t = time.time()

            y.register_hook(
                lambda grad, b=batch: torch.linalg.solve(J[b].transpose(0, 1) + reg*10e-5, grad[:, None])[:, 0]
            )
            out.append(mat(y)[0])
        out = [torch.stack(o, dim=0) for o in zip(*out)]
        return out[0] if len(out) == 1 else tuple(out)

# class OptLayer(nn.Module):
#     """Identical to the example in deep implicity layer tutorial
#     http://implicit-layers-tutorial.org/differentiable_optimization/
#     """

#     def __init__(
#         self, variables, parameters, objective, inequalities, equalities, **cvxpy_opts
#     ):
#         super().__init__()
#         self.variables = variables
#         self.parameters = parameters
#         self.objective = objective
#         self.inequalities = inequalities
#         self.equalities = equalities
#         self.cvxpy_opts = cvxpy_opts

#         # create the cvxpy problem with objective, inequalities, equalities
#         self.cp_inequalities = [
#             ineq(*variables, *parameters) <= 0 for ineq in inequalities
#         ]
#         self.cp_equalities = [eq(*variables, *parameters) == 0 for eq in equalities]
#         self.problem = cp.Problem(
#             cp.Minimize(objective(*variables, *parameters)),
#             self.cp_inequalities + self.cp_equalities,
#         )

#     def forward(self, *batch_params, flag=False):
#         out, J = [], []
#         # solve over minibatch by just iterating
#         for batch in range(batch_params[0].shape[0]):
#             # solve the optimization problem and extract solution + dual variables
#             params = [p[batch] for p in batch_params]
#             with torch.no_grad():
#                 for i, p in enumerate(self.parameters):
#                     p.value = params[i].double().numpy()
#                 self.problem.solve(**self.cvxpy_opts)
#                 z = [torch.tensor(v.value).type_as(params[0]) for v in self.variables]
#                 lam = [
#                     torch.tensor(c.dual_value).type_as(params[0])
#                     for c in self.cp_inequalities
#                 ]
#                 nu = [
#                     torch.tensor(c.dual_value).type_as(params[0])
#                     for c in self.cp_equalities
#                 ]

#             # convenience routines to "flatten" and "unflatten" (z,lam,nu)
#             def vec(z, lam, nu):
#                 return torch.cat([a.view(-1) for b in [z, lam, nu] for a in b])

#             def mat(x):
#                 sz = [0] + list(
#                     accumulate([a.numel() for b in [z, lam, nu] for a in b])
#                 )
#                 val = [x[a:b] for a, b in zip(sz, sz[1:])]
#                 return (
#                     [val[i].view_as(z[i]) for i in range(len(z))],
#                     [val[i + len(z)].view_as(lam[i]) for i in range(len(lam))],
#                     [val[i + len(z) + len(lam)].view_as(nu[i]) for i in range(len(nu))],
#                 )

#             # computes the KKT residual
#             def kkt(z, lam, nu, *params):
#                 g = [ineq(*z, *params) for ineq in self.inequalities]
#                 dnu = [eq(*z, *params) for eq in self.equalities]
#                 L = (
#                     self.objective(*z, *params)
#                     + sum((u * v).sum() for u, v in zip(lam, g))
#                     + sum((u * v).sum() for u, v in zip(nu, dnu))
#                 )
#                 dz = autograd.grad(L, z, create_graph=True)
#                 dlam = [lam[i] * g[i] for i in range(len(lam))]
#                 return dz, dlam, dnu

#             # compute residuals and re-engage autograd tape
#             y = vec(z, lam, nu)
#             if flag:
#                 out.append(mat(y)[0])
#                 continue
#             y = y - vec(
#                 *kkt(
#                     [z_.clone().detach().requires_grad_() for z_ in z], lam, nu, *params
#                 )
#             )

#             # compute jacobian and backward hook
#             J.append(
#                 autograd.functional.jacobian(lambda x: vec(*kkt(*mat(x), *params)), y)
#             )
#             y.register_hook(
#                 lambda grad, b=batch: torch.solve(grad[:, None], J[b].transpose(0, 1))[
#                     0
#                 ][:, 0]
#             )
#             out.append(mat(y)[0])
#         out = [torch.stack(o, dim=0) for o in zip(*out)]
#         return out[0] if len(out) == 1 else tuple(out)




def data_generator(
    c1_value,
    c2_value,
    upperbound_p,
    lowerbound_p,
    upperbound_e,
    lowerbound_e,
    initial_e,
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
    index = 0
    for i in range(N):
        price = np.array(price_hist.RTP[i * 288 : (i + 1) * 288])
        if len(price) != 288 or min(price) < 0:
            continue
        price = np.mean(price.reshape(24,-1),axis=1)
        # price = np.array(price_hist.DAP[i * 288 : (i + 1) * 288 : 12])

        c1 = c1_value
        c2 = c2_value
        P1 = upperbound_p
        P2 = lowerbound_p
        E1 = upperbound_e
        E2 = lowerbound_e
        e0 = initial_e
        eta = efficiency
        # eps = leak

        # define the variable and objective function
        p = cp.Variable(T)
        d = cp.Variable(T)
        f = -price @ (d - p) + c1 * cp.sum(d) + c2 * cp.sum_squares(d)
        # define constraints
        cons = [
            p <= np.ones(T,) * P1,
            p >= np.ones(T,) * P2,
            d <= np.ones(T,) * P1,
            d >= np.ones(T,) * P2,
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) <= np.ones(T,) * (E1 - e0),
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) >= np.ones(T,) * (E2 - e0),
            # np.tril(np.ones((T, T))) @ (eta * p - d / eta) - np.arange(eps, (T+1)*eps, eps) <= np.ones(T,) * (E1 - e0),
            # np.tril(np.ones((T, T))) @ (eta * p - d / eta) - np.arange(eps, (T+1)*eps, eps) >= np.ones(T,) * (E2 - e0),
        ]

        cp.Problem(cp.Minimize(f), cons).solve(
            solver="GUROBI",
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=1000000000,
        )

        df_price[index, :] = price.T
        df_d[index, :] = d.value
        df_p[index, :] = p.value

        index = index + 1

    return df_price, df_d, df_p

def data_generator_val(
    c1_value,
    c2_value,
    upperbound_p,
    lowerbound_p,
    upperbound_e,
    lowerbound_e,
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

    df_d = np.zeros((N, T))
    df_p = np.zeros((N, T))
    index = 0
    dates = np.random.choice(365,N,replace=False)
    for i in range(N):
        price = price_hist[i]

        # price = np.array(price_hist.DAP[i * 288 : (i + 1) * 288 : 12])

        c1 = c1_value
        c2 = c2_value
        P1 = upperbound_p
        P2 = lowerbound_p
        E1 = upperbound_e
        E2 = lowerbound_e
        eta = efficiency
        # eps = leak

        # define the variable and objective function
        p = cp.Variable(T)
        d = cp.Variable(T)
        f = -price @ (d - p) + c1 * cp.sum(d) + c2 * cp.sum_squares(d)
        # define constraints
        cons = [
            p <= np.ones(T,) * P1,
            p >= np.ones(T,) * P2,
            d <= np.ones(T,) * P1,
            d >= np.ones(T,) * P2,
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) <= np.ones(T,) * E1 ,
            np.tril(np.ones((T, T))) @ (eta * p - d / eta) >= np.ones(T,) * E2 ,
        ]

        cp.Problem(cp.Minimize(f), cons).solve(
            solver="GUROBI",
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=1000000000,
        )

        df_d[index, :] = d.value
        df_p[index, :] = p.value

        index = index + 1

    return df_d, df_p
