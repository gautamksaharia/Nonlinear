# Bright Soliton Evolution under the Nonlinear Schrödinger Equation (NLSE)

 the evolution of a **bright soliton** governed by the **1D focusing Nonlinear Schrödinger Equation (NLSE)** using the **split-step Fourier method** in Python.

---

## 🧠 Equation

\[
i \frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + 2 |u|^2 u = 0
\]

---

## 💡 Analytical Solution: Bright Soliton

\[
u(x,t) = \eta \ \text{sech}(\eta(x - 2\xi t - x_0)) \cdot e^{i(\xi x - (\xi^2 - \eta^2)t + \phi_0)}
\]

Where:
- `η`: amplitude
- `ξ`: velocity
- `x₀`: initial position
- `φ₀`: initial phase

---

## 🧮 Numerical Method: Split-Step Fourier

We split the equation into two parts:

### Linear Part:
\[
u \leftarrow e^{-i k^2 \Delta t} u
\]

Handled in **Fourier space** using FFT.

### Nonlinear Part:
\[
u \leftarrow e^{2i |u|^2 \Delta t} u
\]

Handled in **real space**.

---

## 🧾 Python Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1024           # Number of spatial points
L = 20             # Domain size [-L/2, L/2]
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)

T = 10.0           # Total simulation time
dt = 0.01          # Time step
n_steps = int(T / dt)

# Soliton parameters
eta = 1.0          # Amplitude
xi = 0.1           # Velocity
x0 = 0.0           # Initial position
phi0 = 0.0         # Initial phase

# Initial condition: bright soliton at t=0
psi0 = eta * 1/(np.cosh(eta * (x - x0))) * np.exp(1j * (xi * x + phi0))

# Fourier wavenumbers
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# Split-step Fourier method
def split_step_nlse(psi0, dt, n_steps):
    psi = psi0.copy()
    history = [psi.copy()]
    
    exp_L = np.exp(-1j * k**2 * dt)  # Linear evolution operator
    
    for _ in range(n_steps):
        # Nonlinear step
        psi = psi * np.exp(1j * 2 * np.abs(psi)**2 * dt)
        # Linear step in Fourier space
        psi = np.fft.ifft(exp_L * np.fft.fft(psi))
        history.append(psi.copy())
        
    return np.array(history)

# Run simulation
psi_history = split_step_nlse(psi0, dt, n_steps)

# Visualization
plt.plot(x, np.abs(psi_history[0])**2, label="init")
plt.plot(x, np.abs(psi_history[-1])**2, label="final")
plt.xlabel('x')
plt.ylabel('|ψ(x,t)|²')
plt.legend()
plt.title('Bright Soliton Evolution (NLSE)')
plt.show()
