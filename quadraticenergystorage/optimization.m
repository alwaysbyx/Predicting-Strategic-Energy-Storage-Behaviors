clear;
addpath(genpath('/Users/bb/Desktop/UCSD/2022Winter/ECE285/yalmip/YALMIP-master'))
addpath(genpath('/Users/bb/mosek/mosek/9.3/toolbox/r2015a'))

N = 5;
load('dataset/data_N365_0.mat');
price = price(11:10+N,:)';
price = [price; -price];

T = 24;
G = sdpvar(6*T,2*T);
h = sdpvar(6*T,1);
y = sdpvar(2*T, N);
Gy = sdpvar(6*T, N);
mu = sdpvar(6*T, N);
eta2 = sdpvar(1); % eta square
E1eta = sdpvar(1);
E2eta = sdpvar(1);
a1 = sdpvar(1);
a2 = sdpvar(1);

% constraints for G
constraints = [G(2*T+1:4*T, :) == eye(2*T), G(4*T+1:6*T, :) == -eye(2*T)];
constraints = [constraints, G(1:T,1:T) == eta2 * tril(ones(T))];
constraints = [constraints, G(1:T,T+1:2*T) == - tril(ones(T))];
constraints = [constraints, G(T+1:2*T,1:T) == - eta2 * tril(ones(T))];
constraints = [constraints, G(T+1:2*T,T+1:2*T) == tril(ones(T))];
% constraints for h
constraints = [constraints, h(1:T) == E1eta, h(1+T:2*T) == E2eta];
constraints = [constraints, h(2*T+1:4*T) == 0.5, h(4*T+1:6*T) == 0];
% constraints general
constraints = [constraints, 0.6 <= eta2 <= 1.0, a2 >= 0, a1 >= 0];
% 10a
constraints = [constraints, price + a1 + a2 * y + G'*mu == 0];
% 10c
constraints = [constraints, G*y == Gy];
for i = 1:N
    constraints = [constraints, diag(mu(:,i))*(Gy(:,i) - h) == 0];
    constraints = [constraints, (Gy(:,i)-h) <= 0];
end
constraints = [constraints, mu >= 0];

true_p = p(11:10+N, :)';
true_d = d(11:10+N, :)';
obj = 0;
for i = 1:N
obj = obj + (true_p(:,i) - y(1:T,i))'*(true_p(:,i) - y(1:T,i));
obj = obj + (true_d(:,i) - y(T+1:2*T,i))'*(true_d(:,i) - y(1+T:2*T,i));
end

%assign(eta2,0.81);
%assign(a1,10);
%assign(a2,10);
%assign(E1eta,0.5*0.9);
%assign(E2eta,0.5*0.9);
%ops = sdpsettings('usex0',1,'solver', 'gurobi', 'gurobi.TimeLimit', 1200); %, 'bmibnb.maxtime', 600);
20+23
ops = sdpsettings('usex0',1,'solver', 'baron', 'baron.maxtime', 3600); 
diagnostics = optimize(constraints, obj, ops);
time = diagnostics.solvertime;
a1 = double(a1);
a2 = double(a2);
eta2 = double(eta2);
E1eta = double(E1eta);
E2eta = double(E2eta);
save(char(string('dataset/data_0_N5_baron.mat')), 'time', 'a1', 'a2', 'eta2', 'E1eta', 'E2eta');



