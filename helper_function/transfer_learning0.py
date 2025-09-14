import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))   # hidden layers
        x = self.layers[-1](x)         # output layer
        return x

model = PINN([1, 8, 8, 8, 1])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# training data
x = torch.linspace(-10,10,100).unsqueeze(1)   # (x,t)

y = 1/torch.cosh(x) + torch.sin(x)  + 0.5*torch.sin(50*x) + torch.cos(0.5*x)
plt.plot(x, y, label="original")

loss_base=[]
for epoch in range(10000):  # small loop
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    loss_base.append(loss.item())
    if epoch % 50==0:
        print(f"Base {epoch}, Loss: {loss.item():.4f}")


plt.plot(x, model(x).detach().numpy(), label="base")

# Save specific layers
torch.save({
    "layer0": model.layers[0].state_dict(),
    "layer1": model.layers[1].state_dict()
}, "partial_layers.pth")

# Save trained weights
#torch.save(model.state_dict(), "base_model.pth")
#new_model.load_state_dict(torch.load("base_model.pth"))  # load all weights
#print("Weights transferred!")

# New model (for transfer learning)
new_model = PINN([1, 8, 8, 1])  # bigger model
# Load saved weights
saved = torch.load("partial_layers.pth")

# Transfer weights into corresponding layers
new_model.layers[0].load_state_dict(saved["layer0"], strict=False)
new_model.layers[1].load_state_dict(saved["layer1"], strict=False)
print("Transferred selected layers successfully!")



optimizer1 = torch.optim.Adam(new_model.parameters(), lr=1e-3)
loss_trans =[]
for epoch in range(700):  # small loop
    optimizer1.zero_grad()
    output = new_model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer1.step()
    loss_trans.append(loss.item())
    if epoch % 50==0:
        
        print(f"Transfer {epoch}, Loss: {loss.item():.4f}")

plt.plot(x, new_model(x).detach().numpy(), label="transfer")
plt.legend()
plt.title("Model Predictions")
plt.show()


plt.plot(loss_base, label="Base PINN")
plt.plot(loss_trans, label="Transfer PINN")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss")
plt.legend()
plt.show()
