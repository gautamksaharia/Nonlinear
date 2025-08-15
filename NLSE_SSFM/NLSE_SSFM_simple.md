# Bright Soliton Evolution under the Nonlinear Schrödinger Equation (NLSE)

 the evolution of a **bright soliton** governed by the **1D focusing Nonlinear Schrödinger Equation (NLSE)** using the **split-step Fourier method** in Python.

---


## 🧠 Equation

$$
i \frac{\partial u}{\partial t} + \frac{\partial^2 u}{\partial x^2} + 2 |u|^2 u = 0
$$

---

## 💡 Analytical Solution: Bright Soliton

$$
u(x,t) = \eta \ \text{sech}(\eta(x - 2\xi t - x_0)) \cdot e^{i(\xi x - (\xi^2 - \eta^2)t + \phi_0)}
$$

Where:
- $\eta$: amplitude
- $\xi$: velocity
- $x_0$: initial position
- $\phi_0$: initial phase

---


## 🧮 Numerical Method: Split-Step Fourier

We split the equation into two parts:

### Linear Part:
$$
u \leftarrow e^{-i k^2 \Delta t} u
$$

Handled in **Fourier space** using FFT.

### Nonlinear Part:
$$
u \leftarrow e^{2i |u|^2 \Delta t} u
$$

Handled in **real space**.

---

