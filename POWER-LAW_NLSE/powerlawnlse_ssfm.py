# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 19:09:05 2025

@author: gauta
"""

"""
Strang split-step pseudospectral solver for i q_t + 1/2 q_xx + |q|^{2p} q = 0
Simulates a 1-soliton initial condition of the sech^{1/p} form and reports
mass, momentum, Hamiltonian and L2 error vs analytic solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift
from scipy.special import gamma
import math
from scipy.integrate import trapezoid
np.random.seed(21)

# ---------------------------
# Parameters (change here)
# ---------------------------
p = 1.0              # nonlinearity exponent (0 < p < 2)
A = 1.0                 # soliton amplitude
v = 1.0                 # soliton velocity
x0 = 0.0               # initial center
sigma0 = 0.0            # initial global phase
L = 40.0                # domain length
N = 2**15               # number of grid points (power of 2)
dt = 0.001            # time step
tmax = 10.0              # final time
save_every = 100  # how often to store/plot (in steps)

# ---------------------------
# Derived parameters & grid
# ---------------------------
dx = L / N
x = np.linspace(-L/2, L/2, N, endpoint=False)
k = 2*np.pi * fftfreq(N, d=dx)  # wave numbers
ik2 = 0.5 * (k**2)               # appears in linear propagator exponent

# Analytical soliton parameters from your formula:
# B = A p sqrt((1+2p)/(2p)), kappa = -v, omega = (B^2 - p^2*kappa^2)/(2 p^2)
B = A**p*np.sqrt(2*p**2/(1+p))
kappa = -v
omega = (B**2 - (p**2) * (kappa**2)) / (2 * p**2)

def soliton_profile(xvals, t=0.0):
    """Analytic 1-soliton q(x,t)."""
    y = xvals - v*t - x0
    amp = A * (1.0/np.cosh(B*y))**(1.0/p)
    phase = np.exp(-1j * kappa * xvals + 1j * omega * t + 1j * sigma0)
    return amp * phase

# Initial condition
q = soliton_profile(x, t=0.0)
q1 = soliton_profile(x, t=10.0)
plt.plot(x, np.abs(q), label="initial")
plt.plot(x, np.abs(q1), label="final")
print(np.max(q))
plt.legend()
plt.show()


# Conserved-quantity functions
def mass(qarr):
    return trapezoid( np.abs(qarr)**2,  x)

def momentum(qarr):
    # M = (i/2) \int (q* q_x - q q*_x) dx = \int Im(q^* q_x) dx
    dqdx = np.gradient(qarr, dx)
    return trapezoid(np.imag(np.conjugate(qarr) * dqdx), x)

def hamiltonian(qarr):
    dqdx = np.gradient(qarr, dx)
    kinetic = 0.5 * trapezoid(np.abs(dqdx)**2, x)
    potential = (1.0/(p + 1.0)) * trapezoid( np.abs(qarr)**(2*p+2), x)
    return kinetic - potential

# Projection onto internal modes: translation mode (d/dx0 q = -q_x), phase mode (i q)
# Use complex L2 inner product <f,g> = \int conj(f) g dx
def inner(u, v):
    return trapezoid(np.conjugate(u) * v, x)


# High-k (radiation) energy fraction: compute energy in |k|>k_cut relative to total
def radiation_fraction(qarr, k_cut_factor):
    qhat = fft(qarr)
    kabs = np.abs(k)
    # choose cutoff relative to characteristic soliton width: use B from analytic
    # fallback: set cutoff as k_cut_factor * (typical B)
    # Here we'll pass B as an argument later; placeholder not used here
    return qhat, kabs  # handled inline where B known



# Time integration (Strang)
nsteps = int(np.round(tmax / dt))
times = []
mass_list = []
mom_list = []
H_list = []
L2err = []
snapshot_times = []
snapshots = []

# initial conserved values
times.append(0.0)
mass_list.append(mass(q))
mom_list.append(momentum(q))
H_list.append(hamiltonian(q))
L2err.append(np.linalg.norm(q - soliton_profile(x, t=0.0)) * np.sqrt(dx))
snapshots.append(q.copy())
snapshot_times.append(0.0)

print("Starting integration: N=%d, dt=%g, nsteps=%d" % (N, dt, nsteps))
# Precompute linear propagator
# Linear ODE: q_t = -i/2 q_xx -> 
# in Fourier qhat_t = i k^2/2 qhat -> exp(i k^2 dt/2)
L_prop_full = np.exp(-1j * ik2 * dt)           # full linear step
L_prop_half = np.exp(1j * ik2 * (dt/2.0))     # half linear step

for n in range(1, nsteps+1):
    # Nonlinear half-step: q -> q * exp(-i |q|^{2p} dt/2)
    q = q * np.exp(1j * (np.abs(q)**(2*p)) * (dt/2.0))

    # Linear full-step in Fourier
    qhat = fft(q)
    qhat = qhat * L_prop_full
    q = ifft(qhat)

    # Nonlinear half-step
    q = q * np.exp(1j * (np.abs(q)**(2*p)) * (dt/2.0))
    


    t = n * dt

    # Diagnostics
    if (n % save_every == 0) or (n == nsteps):
        times.append(t)
        mass_list.append(mass(q))
        mom_list.append(momentum(q))
        H_list.append(hamiltonian(q))
        L2err.append(np.linalg.norm(q - soliton_profile(x, t=t)) * np.sqrt(dx))
        snapshots.append(q.copy())
        snapshot_times.append(t)
        print(f"t={t:.3f}  mass={mass_list[-1]:.6f}  mom={mom_list[-1]:.6f}  H={H_list[-1]:.6f}  L2err={L2err[-1]:.3e}")

# ---------------------------
# Plot results
# ---------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(times, mass_list, '-o', label='Mass')
plt.xlabel('t'); plt.ylabel('Mass'); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(times, H_list, '-o', label='Hamiltonian')
plt.xlabel('t'); plt.ylabel('Hamiltonian'); plt.grid(True)
plt.suptitle('Conserved quantities (approx)')
plt.tight_layout()
plt.show()

# snapshots: amplitude and analytic comparison for last snapshot
fig, ax = plt.subplots(1,2, figsize=(12,4))
qnum = np.array(snapshots)
tlast = snapshot_times[-1]

ax[0].plot(x, np.abs(qnum[0]), label='initial')
ax[0].plot(x, np.abs(qnum[-1]), label='final')
ax[0].plot(x, np.abs(soliton_profile(x, t=tlast)), '--', label='analytical')
ax[0].set_xlim(x.min(), x.max())
ax[0].legend(); ax[0].set_title("Amplitude ")

ax[1].plot(x, np.angle(qnum[-1]), label='phase numeric')
ax[1].plot(x, np.angle(soliton_profile(x, t=tlast)), '--', label='phase analytic')
ax[1].set_xlim(x.min(), x.max())
ax[1].legend(); ax[1].set_title("Phase")
plt.show()

# final L2 error
print("Final L2 error:", L2err[-1])
