# -*- coding: utf-8 -*-
"""8.pytorch_PINN_ODE_legendre.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JPgwSkSeZfHHI7bmIoo-zmKa4ahP75qD

#Legendre  Eqaution

$$(1-x^2)\frac{d^2y}{dx^2} - 2 x\frac{dy}{dx}+n(n+1)y=0$$

For n = 2, $$y= \frac{1}{2}(3x^2-1)$$


For n = 3, $$y= \frac{1}{2}(5x^3-3x)$$

PINN to solve PDE
"""

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x_min = -2
x_max = 3
N = 50
np.random.seed(2)

x = np.random.uniform(low=x_min, high=x_max, size=(N,1))
#x = np.linspace(x_min, x_max, N).reshape(N,1)
y_true = 0.5*(5*x**3 -3*x)
xnp = np.linspace(x_min,x_max,10)
ynp = 0.5*(5*xnp**3 -3*xnp)

plt.plot(xnp, ynp, 'r--')
plt.scatter(x, y_true)
plt.xlabel("x axis")
plt.ylabel("y")
plt.show()

"""## Neural Network"""

class Neural_network(nn.Module):
  def __init__(self,neuron_no):
    super(Neural_network, self).__init__()
    self.input_layer = nn.Linear(1,neuron_no)
    self.hidden_layer1 = nn.Linear(neuron_no,neuron_no)
    self.hidden_layer2 = nn.Linear(neuron_no,neuron_no)
    self.hidden_layer3 = nn.Linear(neuron_no,neuron_no)
    self.output_layer = nn.Linear(neuron_no, 1)

  def forward(self, x):
    x = torch.tanh(self.input_layer(x))
    x = torch.tanh(self.hidden_layer1(x))
    x = torch.tanh(self.hidden_layer2(x))
    x = torch.tanh(self.hidden_layer3(x))
    x = self.output_layer(x)
    return x

Net = Neural_network(20)
Net=Net.to(device)

"""# ODE Loss"""

def ode(x, Net):
  x_1 = torch.autograd.Variable( torch.from_numpy(x).float(), requires_grad=True).to(device)
  
  y = Net.forward(x_1)

  y_x = torch.autograd.grad(y.sum(), x_1, create_graph=True)[0]
  y_xx = torch.autograd.grad(y_x.sum(), x_1, create_graph=True)[0]
  ode = (1- x_1**2)*y_xx - 2*x_1*y_x + 12*y
  return ode, y

"""#Loss """

optim = torch.optim.Adam(Net.parameters(), lr=0.001)
LL = nn.MSELoss()

number_of_epoch =10000

loss_value = []
for epoch in range(number_of_epoch):

  
  loss1 = torch.mean(torch.square( Net.forward( torch.tensor([-2.]).to(device) ) - torch.tensor([-17.]).to(device) ))
  loss2 = torch.mean(torch.square( Net.forward( torch.tensor([3.]).to(device) ) - torch.tensor([63.]).to(device) ))
  ode1, _ = ode(x, Net)
  
  Loss_ode = torch.mean(ode1**2)
  total_loss =  0.1*Loss_ode + loss1 + loss2  
  total_loss.backward()    # computing gradients using backward propagation  dL/dw
  optim.step()             # This is equivalent to : Weight_new = weight_old - learing_rate * derivative of Loss w.r.t weight
  optim.zero_grad()       # make the gradient zero
  loss_value.append(total_loss.cpu().detach().numpy())
  with torch.autograd.no_grad():
    if epoch%100==0:
      print(f'epoch:{epoch}, loss:{total_loss.item():.8f}, loss_ode:{Loss_ode.item():.8f}' )

plt.plot(loss_value)

"""## total time need to train"""

#prediction and plot
plt.scatter(x, ode(x, Net)[1].cpu().detach().numpy(), label="approx")
plt.scatter(x, y_true, label="exact", marker='.')
plt.show()

xz = np.linspace(x_min, x_max, 100)
xz1 = xz.reshape(100,1)
xz2 =  torch.autograd.Variable( torch.from_numpy(xz1).float(), requires_grad=True).to(device)
yz2 = Net.forward(xz2)
yz3 = yz2.cpu().detach().numpy()
plt.scatter(xz1, yz3)