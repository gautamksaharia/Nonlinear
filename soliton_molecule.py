
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
Nx = 1024          # spatial grid points
L = 20.0          # half-domain size
dx = 2 * L / Nx
x = np.linspace(-L, L - dx, Nx)

dt = 0.001         # time step
Nt = 50000         # number of time steps
output_every = 50

epsilon = 0.15     # non-integrability strength   
epsilon_tod =0.00   # non-integrability strength TOD
v = 0.5            # relative velocity
d = 10.0           # initial separation
delta_phi = 0.10    # relative phase

# -----------------------------
# Fourier space
# -----------------------------
k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)


# -----------------------------
# Absorbing boundary mask
# -----------------------------
L_abs = 70.0
alpha = 0.05
mask = np.ones_like(x)
idx = np.abs(x) > L_abs
mask[idx] = np.exp(-alpha * (np.abs(x[idx]) - L_abs)**2)

# -----------------------------
# Soliton initial condition
# -----------------------------
def soliton(x, x0, v, phi):
    return (1.0 / np.cosh(x - x0)) * np.exp(1j * (v * x + phi))

psi1 = soliton(x, -d/2,  v/2, 0.0)
psi2 = soliton(x,  d/2, -v/2, delta_phi)
psi = psi1 + psi2

plt.plot(abs(psi))
plt.show()
# -----------------------------
# Diagnostics storage
# -----------------------------
times = []
energies = []
mass = []
momentum = []

separations = []
psi_history=[]


# -----------------------------
# Storage for phase tracking
# -----------------------------

phase_track = []

# -----------------------------
# Energy functional
# -----------------------------
def energy(psi):
    psi_x = np.fft.ifft(1j * k * np.fft.fft(psi))
    energy = np.trapezoid(0.5 * np.abs(psi_x)**2 - 0.5 * np.abs(psi)**4, x)
    mass = np.trapezoid(np.abs(psi)**2, x)
    momentum = 0.5*1j*(np.trapezoid(psi*np.conjugate(psi_x) -psi_x*np.conjugate(psi) , x))
    return energy, mass, momentum

# -----------------------------
# Center-of-mass estimator
# -----------------------------
def soliton_centers(psi):
    density = np.abs(psi)**2
    left = x < 0
    right = x > 0
    xL = np.trapezoid(x[left] * density[left], x[left]) / np.trapezoid(density[left], x[left])
    xR = np.trapezoid(x[right] * density[right], x[right]) / np.trapezoid(density[right], x[right])
    return xL, xR

# -----------------------------
# Time evolution
# -----------------------------
for n in range(Nt):

    
    L_half = np.exp(-1j * (k**2 / 2) * (dt/2) - 1j*epsilon_tod*k**3 *(dt/2))
    
    # Linear half-step
    psi_hat = np.fft.fft(psi)
    psi_hat *= L_half
    psi = np.fft.ifft(psi_hat)

    # Nonlinear full-step (cubic + quintic)
    nonlinear_phase = np.exp(1j * dt * np.abs(psi)**2    + 1j * dt * epsilon * np.abs(psi)**4    )
    psi *= nonlinear_phase

    # Linear half-step
    psi_hat = np.fft.fft(psi)
    psi_hat *= L_half
    psi = np.fft.ifft(psi_hat)

    # Absorbing boundaries
    psi *= mask

    # Diagnostics
    if n % output_every == 0:
        t = n * dt
        E, mas, momentm = energy(psi)
        xL, xR = soliton_centers(psi)

        times.append(t)
        energies.append(E)
        mass.append(mas)
        momentum.append(momentm)
        separations.append(xR - xL)
        psi_history.append(psi)
        
        abspsi = np.abs(psi)
        idx = np.argmax(abspsi)
        phase = np.angle(psi[idx])
        phase_track.append(phase)

        print(f"t = {t:.2f}, E = {E:.6f}, separation = {xR - xL:.3f}")

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(times, energies)
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.title("Energy evolution")
plt.ylim(-5,5)

plt.subplot(1, 2, 2)
plt.plot(times, separations)
plt.xlabel("Time")
plt.ylabel("Soliton separation")
plt.title("Collision outcome")

plt.tight_layout()
plt.show()


plt.imshow(abs(np.array(psi_history)))
plt.show()









times = np.array(times)
phase_track = np.unwrap(np.array(phase_track))

# Remove trivial phase evolution
eta =1
omega1 = eta**2 - v**2 / 4
phase_corrected = phase_track + omega1 * times

# -----------------------------
# Phase shift calculation
# -----------------------------
phi_before = np.mean(phase_corrected[times < 10.5])
phi_after  = np.mean(phase_corrected[times > 25.5])

Delta_phi = phi_after - phi_before

print(f"Numerical phase shift Δφ = {Delta_phi:.6f}")



# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(times, phase_corrected, lw=2)
plt.axvline(1.5, ls='--', color='k')
plt.axvline(4.5, ls='--', color='k')
plt.xlabel("Time")
plt.ylabel("Corrected soliton phase")
plt.title("Numerical soliton phase shift")
plt.tight_layout()
plt.show()




# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(times, np.array(mass), lw=2)
plt.xlabel("Time")
plt.ylabel("mass")
plt.title("norm mass")
# Disable both the offset and scientific notation
plt.ylim(0,6)
plt.tight_layout()
plt.show()


















