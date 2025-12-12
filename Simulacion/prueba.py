# train_phy_nn_2dof.py
# Requisitos: python3.8+, torch, numpy, matplotlib
# pip install torch numpy matplotlib

import math, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def true_dynamics(q, qd, tau, params):
    q = np.array(q); qd = np.array(qd); tau = np.array(tau)
    q1 = q[...,0]; q2 = q[...,1]; q1d = qd[...,0]; q2d = qd[...,1]
    m1, m2 = params['m1'], params['m2']
    lc1, lc2 = params['lc1'], params['lc2']
    I1, I2 = params['I1'], params['I2']; gconst = params['g']
    M11 = I1 + I2 + m2*params['l1']**2 + 2*m2*params['l1']*lc2*np.cos(q2) + params['m1']*lc1**2 + m2*lc2**2
    M12 = I2 + m2*params['l1']*lc2*np.cos(q2) + m2*lc2**2
    M21 = M12.copy()
    M22 = (I2 + m2*lc2**2) * np.ones_like(M11)
    h = -m2*params['l1']*lc2*np.sin(q2)
    C1 = h*(2*q1d*q2d + q2d**2); C2 = h*( - q1d**2 )
    g1 = (params['m1']*lc1 + m2*params['l1'])*gconst*np.cos(q1) + m2*lc2*gconst*np.cos(q1+q2)
    g2 = m2*lc2*gconst*np.cos(q1+q2)
    rhs1 = tau[...,0] - C1 - g1; rhs2 = tau[...,1] - C2 - g2
    det = M11*M22 - M12*M21
    det = np.where(np.abs(det)<1e-8, 1e-8, det)
    inv11 = M22/det; inv12 = -M12/det; inv21 = -M21/det; inv22 = M11/det
    qdd1 = inv11*rhs1 + inv12*rhs2; qdd2 = inv21*rhs1 + inv22*rhs2
    qdd = np.stack([qdd1, qdd2], axis=-1)
    Mstack = np.stack([M11, M12, M21, M22], axis=-1)
    Cstack = np.stack([C1, C2], axis=-1)
    gstack = np.stack([g1, g2], axis=-1)
    return qdd, Mstack, Cstack, gstack

# robot params
params = {'m1':1.2,'m2':0.8,'l1':0.6,'l2':0.4,'lc1':0.3,'lc2':0.2,'I1':0.02,'I2':0.01,'g':9.81}

def gen_traj(n_samples):
    t = np.linspace(0, 6, n_samples)
    q1 = 0.8*np.sin(0.9*t) + 0.2*np.sin(2.1*t + 0.5)
    q2 = 0.6*np.sin(1.3*t + 0.2) + 0.15*np.sin(3.0*t)
    q1d = np.gradient(q1, t); q2d = np.gradient(q2, t)
    q1dd = np.gradient(q1d, t); q2dd = np.gradient(q2d, t)
    q = np.stack([q1,q2],axis=-1); qd = np.stack([q1d,q2d],axis=-1); qdd = np.stack([q1dd,q2dd],axis=-1)
    return q, qd, qdd

# dataset size (ajusta según CPU/GPU)
N = 3000
q, qd, qdd = gen_traj(N+100)
tau_nom = np.zeros_like(q)
for i in range(q.shape[0]):
    qi = q[i]; qdi = qd[i]; qddi = qdd[i]
    q1,q2 = qi[0], qi[1]; q1d,q2d = qdi[0], qdi[1]
    M11 = params['I1'] + params['I2'] + params['m2']*params['l1']**2 + 2*params['m2']*params['l1']*params['lc2']*np.cos(q2) + params['m1']*params['lc1']**2 + params['m2']*params['lc2']**2
    M12 = params['I2'] + params['m2']*params['l1']*params['lc2']*np.cos(q2) + params['m2']*params['lc2']**2
    M21 = M12; M22 = params['I2'] + params['m2']*params['lc2']**2
    h = -params['m2']*params['l1']*params['lc2']*np.sin(q2)
    C1 = h*(2*q1d*q2d + q2d**2); C2 = h*( - q1d**2 )
    g1 = (params['m1']*params['lc1'] + params['m2']*params['l1'])*params['g']*np.cos(q1) + params['m2']*params['lc2']*params['g']*np.cos(q1+q2)
    g2 = params['m2']*params['lc2']*params['g']*np.cos(q1+q2)
    qddi_vec = qddi
    tau1 = M11*qddi_vec[0] + M12*qddi_vec[1] + C1 + g1
    tau2 = M21*qddi_vec[0] + M22*qddi_vec[1] + C2 + g2
    tau_nom[i,0] = tau1; tau_nom[i,1] = tau2

b1,b2 = 0.08, 0.06
tau_nc = - np.stack([b1*qd[:,0] + 0.02*np.tanh(3*qd[:,0]), b2*qd[:,1] + 0.015*np.tanh(4*qd[:,1])], axis=-1)
tau = tau_nom + tau_nc + 0.02*np.random.randn(*tau_nom.shape)
qdd_meas, Mstack2, Cstack2, gstack2 = true_dynamics(q, qd, tau, params)

# dataset (q, qd, tau) -> qdd_meas
X = np.concatenate([q, qd, tau], axis=-1); Y = qdd_meas
N_total = X.shape[0]; idx = np.arange(N_total); np.random.shuffle(idx)
trainN = int(0.8*N_total); train_idx = idx[:trainN]; test_idx = idx[trainN:]
Xtrain = X[train_idx]; Ytrain = Y[train_idx]; Xtest = X[test_idx]; Ytest = Y[test_idx]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Xtrain_t = torch.tensor(Xtrain, dtype=torch.float32).to(device); Ytrain_t = torch.tensor(Ytrain, dtype=torch.float32).to(device)
Xtest_t = torch.tensor(Xtest, dtype=torch.float32).to(device); Ytest_t = torch.tensor(Ytest, dtype=torch.float32).to(device)
train_ds = TensorDataset(Xtrain_t, Ytrain_t); test_ds = TensorDataset(Xtest_t, Ytest_t)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True); test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

print("Dataset: train", len(train_ds), "test", len(test_ds), "device", device)

class SPDModule(nn.Module):
    def __init__(self, n, hidden=64):
        super().__init__(); self.n = n
        self.fc = nn.Sequential(nn.Linear(n, hidden), nn.Tanh(), nn.Linear(hidden, n*n))
    def forward(self, q):
        B = q.shape[0]; out = self.fc(q); L = out.view(B, self.n, self.n); L = torch.tril(L)
        eps = 1e-3; I = torch.eye(self.n, device=q.device).unsqueeze(0).repeat(B,1,1)
        M = torch.bmm(L, L.transpose(-1,-2)) + eps * I; return M

class PHYNN2DOF(nn.Module):
    def __init__(self, hidden=128):
        super().__init__(); self.n = 2
        self.M_net = SPDModule(self.n, hidden=hidden)
        self.C_net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 4))
        self.g_net = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.tau_nc_net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, 2))
    def forward(self, q, qd, u):
        M = self.M_net(q); Cflat = self.C_net(torch.cat([q, qd], dim=-1)); C = Cflat.view(-1,2,2)
        g = self.g_net(q); tau_nc = self.tau_nc_net(torch.cat([q, qd, u], dim=-1))
        rhs = u - torch.bmm(C, qd.unsqueeze(-1)).squeeze(-1) - g - tau_nc
        qdd = torch.linalg.solve(M, rhs.unsqueeze(-1)).squeeze(-1)
        return qdd, M, C, g, tau_nc

model = PHYNN2DOF(hidden=128).to(device)

def loss_fn(model, x_batch, y_batch, alpha_phys=10.0):
    q = x_batch[:,0:2]; qd = x_batch[:,2:4]; u = x_batch[:,4:6]
    qdd_pred, M, C, g, tau_nc = model(q, qd, u)
    mse_data = torch.mean((qdd_pred - y_batch)**2)
    resid = torch.bmm(M, qdd_pred.unsqueeze(-1)).squeeze(-1) + torch.bmm(C, qd.unsqueeze(-1)).squeeze(-1) + g + tau_nc - u
    phys_loss = torch.mean(resid**2)
    loss = mse_data + alpha_phys * phys_loss
    return loss, mse_data.item(), phys_loss.item()

opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
n_epochs = 40
history = {'loss': [], 'mse_data': [], 'phys_loss': [], 'val_mse': []}

for ep in range(n_epochs):
    model.train()
    running_loss = running_mse = running_phys = 0.0
    for xb, yb in train_loader:
        opt.zero_grad(); loss, mse_d, phys_l = loss_fn(model, xb, yb, alpha_phys=10.0)
        loss.backward(); opt.step()
        running_loss += loss.item() * xb.shape[0]; running_mse += mse_d * xb.shape[0]; running_phys += phys_l * xb.shape[0]
    running_loss /= len(train_loader.dataset); running_mse /= len(train_loader.dataset); running_phys /= len(train_loader.dataset)
    model.eval(); val_mse = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            q = xb[:,0:2]; qd = xb[:,2:4]; u = xb[:,4:6]
            qdd_pred, *_ = model(q, qd, u)
            val_mse += torch.sum((qdd_pred - yb)**2).item()
        val_mse /= len(test_loader.dataset)
    history['loss'].append(running_loss); history['mse_data'].append(running_mse); history['phys_loss'].append(running_phys); history['val_mse'].append(val_mse)
    if ep % 5 == 0 or ep == n_epochs-1:
        print(f"Ep {ep:03d} | loss {running_loss:.6f} | mse_train {running_mse:.6e} | phys {running_phys:.6e} | val_mse {val_mse:.6e}")

# evaluation
model.eval()
with torch.no_grad():
    q = Xtest_t[:,0:2]; qd = Xtest_t[:,2:4]; u = Xtest_t[:,4:6]
    qdd_pred, *_ = model(q, qd, u)
    test_rmse = torch.sqrt(torch.mean((qdd_pred - Ytest_t)**2, dim=0))
    print("Test RMSE per joint (rad/s^2):", test_rmse.cpu().numpy())

# plot quick results
nsamp = 200
xs = Xtest[:nsamp]; ys = Ytest[:nsamp]
with torch.no_grad():
    xt = torch.tensor(xs, dtype=torch.float32).to(device)
    q = xt[:,0:2]; qd = xt[:,2:4]; u = xt[:,4:6]
    qddp, *_ = model(q, qd, u); qddp = qddp.cpu().numpy()

plt.figure(); plt.plot(history['loss']); plt.title("Loss de entrenamiento"); plt.xlabel("época"); plt.grid(True); plt.show()
plt.figure(); plt.plot(ys[:,0], label='qdd1_true'); plt.plot(qddp[:,0], label='qdd1_pred', linestyle='--'); plt.legend(); plt.title("qdd1 true vs pred"); plt.show()
plt.figure(); plt.plot(ys[:,1], label='qdd2_true'); plt.plot(qddp[:,1], label='qdd2_pred', linestyle='--'); plt.legend(); plt.title("qdd2 true vs pred"); plt.show()

# save model and summary
os.makedirs('out', exist_ok=True)
model_path = 'out/phy_nn_2dof.pth'
torch.save({'model_state_dict': model.state_dict(), 'params': params}, model_path)
summary = {'train_samples': len(train_ds), 'test_samples': len(test_ds), 'n_epochs': n_epochs, 'final_train_loss': history['loss'][-1], 'final_val_mse': history['val_mse'][-1], 'test_rmse_joint': test_rmse.cpu().numpy().tolist(), 'model_file': model_path}
with open('out/training_summary.json','w') as f: json.dump(summary, f, indent=2)
print("Modelo y resumen guardados en carpeta 'out'")
