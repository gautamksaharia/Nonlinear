import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Simulation parameters
# ---------------------------
Nt = 2**10                 # number of time points
Tmax = 20.0                # time window
dt = 2*Tmax/Nt             # time step
t = np.linspace(-Tmax, Tmax-dt, Nt)

dz = 0.01                  # propagation step
Nz = 500                   # number of steps
z_points = np.arange(0, Nz*dz, dz)

# Frequency grid
dw = np.pi/Tmax
w = np.fft.fftfreq(Nt, d=dt/(2*np.pi))  # angular frequency grid

# Fiber parameters (dimensionless normalized units)
beta2 = -1.0       # anomalous dispersion
gamma = 1.0        # nonlinearity
f_R = 0.18         # Raman fraction (typical for silica)
tau1 = 12.2e-3     # Raman response time (normalized)
tau2 = 32e-3

# Raman response function in time domain (normalized approximation)
def raman_response(t):
    return ( (tau1**2 + tau2**2)/(tau1*tau2**2) ) * np.exp(-t/tau2)*np.sin(t/tau1)*(t>0)

Rt = raman_response(t)
Rw = np.fft.fft(np.fft.ifftshift(Rt))*dt  # FFT of response

# ---------------------------
# Initial condition: fundamental soliton
# ---------------------------
T0 = 1.0  # soliton width
A0 = 1.0/T0  # amplitude for fundamental soliton in normalized NLSE
u0 = A0 * (1/np.cosh(t/T0))

# Storage arrays
Omega = []
spec_evolution = []

# ---------------------------
# Split-step Fourier method with Raman
# ---------------------------
u = u0.copy()
linear_operator = np.exp(0.5j*beta2*(w**2)*dz)  # dispersion half step

for zi in z_points:
    # store spectrum
    U = np.fft.fft(u)
    spec = np.abs(U)**2
    spec_evolution.append(spec)

    # compute mean frequency (Omega)
    Omega_z = np.sum(w*spec)/np.sum(spec)
    Omega.append(Omega_z)

    # Nonlinear step (with Raman convolution)
    It = np.abs(u)**2
    conv = np.fft.ifft(np.fft.fft(It)*Rw).real  # Raman convolution
    NL = np.exp(1j*gamma*( (1-f_R)*It + f_R*conv )*dz)
    u = u * NL

    # Linear step
    U = np.fft.fft(u)
    U = U * linear_operator
    u = np.fft.ifft(U)

# Convert lists to arrays
Omega = np.array(Omega)
spec_evolution = np.array(spec_evolution)

# ---------------------------
# Plot results
# ---------------------------

# Central frequency shift vs z
plt.figure(figsize=(6,4))
plt.plot(z_points, Omega, 'b-')
plt.xlabel("Propagation distance z")
plt.ylabel("Central frequency Ω(z)")
plt.title("Soliton self-frequency shift (with Raman)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Spectral evolution heatmap
plt.figure(figsize=(7,5))
extent=[w.min(), w.max(), z_points.min(), z_points.max()]
plt.imshow(10*np.log10(spec_evolution+1e-12), aspect='auto', extent=extent, 
           origin='lower', cmap='jet')
plt.colorbar(label="Spectral intensity (dB)")
plt.xlabel("Frequency ω")
plt.ylabel("Propagation distance z")
plt.title("Spectral evolution with Raman-induced SSFS")
plt.tight_layout()
plt.show()
