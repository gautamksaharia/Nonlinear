# PINN for Inverse NLSE using 1-soliton analytic data

import torch
import torch.nn as nn
import numpy as np

# PINN Model
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 2]):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers_list = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x, t):
        X = torch.cat((x, t), dim=1)
        for i in range(len(self.layers_list)-1):
            X = self.activation(self.layers_list[i](X))
        return self.layers_list[-1](X)


# NLSE Residuals
def NLSE_residuals(model, x, t, lambda_param):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = model(x, t)
    ur, ui = u[:,0:1], u[:,1:2]

    ur_t = torch.autograd.grad(ur, t, torch.ones_like(ur), create_graph=True)[0]
    ui_t = torch.autograd.grad(ui, t, torch.ones_like(ui), create_graph=True)[0]

    ur_x = torch.autograd.grad(ur, x, torch.ones_like(ur), create_graph=True)[0]
    ui_x = torch.autograd.grad(ui, x, torch.ones_like(ui), create_graph=True)[0]

    ur_xx = torch.autograd.grad(ur_x, x, torch.ones_like(ur_x), create_graph=True)[0]
    ui_xx = torch.autograd.grad(ui_x, x, torch.ones_like(ui_x), create_graph=True)[0]

    abs_u2 = ur**2 + ui**2

    R1 = ur_t + ui_xx + lambda_param * abs_u2 * ui
    R2 = -ui_t + ur_xx + lambda_param * abs_u2 * ur
    return R1, R2


# 1-Soliton Analytic Solution
def soliton(x, t):
    eta = 1.0
    xi = 0.0
    x0 = 0.0
    phi0 = 0.0

    phase = xi * x + (eta**2 - xi**2)*t + phi0
    envelope = eta /torch.cosh(eta * (x - 2*xi*t - x0))

    u_r = envelope * torch.cos(phase)
    u_i = envelope * torch.sin(phase)
    return u_r, u_i


# Generate Measurement Data from Soliton
def generate_soliton_data():
    N_data = 200
    N_f = 2000

    x_min, x_max = -5.0, 5.0
    t_min, t_max = 0.0, 1.0

    # Measurement locations
    x_data = torch.rand(N_data, 1) * (x_max - x_min) + x_min
    t_data = torch.rand(N_data, 1) * (t_max - t_min) + t_min

    # Evaluate soliton
    ur, ui = soliton(x_data, t_data)
    u_data = torch.cat((ur, ui), dim=1)

    # Physics collocation points
    x_f = torch.rand(N_f, 1) * (x_max - x_min) + x_min
    t_f = torch.rand(N_f, 1) * (t_max - t_min) + t_min

    return x_data, t_data, u_data, x_f, t_f


# Train PINN to Recover λ
def train():
    # Create model and unknown lambda parameter
    model = PINN()
    lambda_param = torch.tensor([2.40], requires_grad=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [lambda_param],
        lr=1e-3
    )

    # Generate soliton data
    x_data, t_data, u_data, x_f, t_f = generate_soliton_data()

    for it in range(1, 5001):
        # Data loss
        u_pred = model(x_data, t_data)
        loss_data = torch.mean((u_pred - u_data)**2)

        # PDE loss
        R1, R2 = NLSE_residuals(model, x_f, t_f, lambda_param)
        loss_pde = torch.mean(R1**2 + R2**2)

        # Total loss
        loss = loss_data + 1e-2 * loss_pde

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 50 == 0:
            print(f"[{it:05d}] Loss = {loss.item():.3e} | λ = {lambda_param.item():.6f}")

    print("Training complete.")
    print("Recovered λ =", float(lambda_param.item()))


if __name__ == "__main__":
    train()
