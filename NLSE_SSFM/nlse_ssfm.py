import numpy as np
import matplotlib.pyplot as plt

# Constants
p0 = 5e-3  # Watt
gamma = 1.2  # 1/km
beta2 = -21.6e-24  # s^2/km
l = 500  # km
t0_2 = abs(beta2) / (gamma * p0)  # s^2
t0 = np.sqrt(t0_2)  # s
Ts = 1e-12
t = np.arange(-1e-9, 1e-9, Ts)
fs = 1 / Ts
dz = 0.1  # km
num_of_steps = round(l / dz)
C = 0  # chirp

def d_operator(input_signal, beta2, dz, freq):
    D_freq = np.exp(1j * beta2 * dz * (2 * np.pi * freq)**2 / 2)
    return input_signal * D_freq

def n_operator(input_signal, gamma, dz):
    N = np.exp(1j * 8 * gamma * dz * np.abs(input_signal)**2 / 9)
    return input_signal * N

# Input Pulse (Soliton)
input_pulse = np.sqrt(p0) * np.cosh(t / t0)**-1
df = fs / len(t)
freq = np.fft.fftfreq(len(t), Ts)

# Symmetric Method
output_symmetric = np.fft.fftshift(np.fft.fft(input_pulse))
for _ in range(num_of_steps):
    output_symmetric = d_operator(output_symmetric, beta2, dz / 2, freq)
    output_symmetric = np.fft.ifft(np.fft.ifftshift(output_symmetric))
    output_symmetric = n_operator(output_symmetric, gamma, dz)
    output_symmetric = np.fft.fftshift(np.fft.fft(output_symmetric))
    output_symmetric = d_operator(output_symmetric, beta2, dz / 2, freq)
output_symmetric = np.fft.ifft(np.fft.ifftshift(output_symmetric))

# Asymmetric Method
output_asymmetric = np.fft.fftshift(np.fft.fft(input_pulse))
for _ in range(num_of_steps):
    output_asymmetric = d_operator(output_asymmetric, beta2, dz, freq)
    output_asymmetric = np.fft.ifft(np.fft.ifftshift(output_asymmetric))
    output_asymmetric = n_operator(output_asymmetric, gamma, dz)
    output_asymmetric = np.fft.fftshift(np.fft.fft(output_asymmetric))
output_asymmetric = np.fft.ifft(np.fft.ifftshift(output_asymmetric))

# Plotting Soliton Pulse Evolution
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t * 1e9, input_pulse, label="Input Pulse", color="red", linestyle="-")
plt.plot(t * 1e9, np.abs(output_symmetric), label="Output Pulse", color="blue", linestyle="-.")
plt.grid(which='minor')
plt.title("Pulse Evolution (Soliton Pulse) (Symmetric)")
plt.xlabel("Time (ns)")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t * 1e9, input_pulse, label="Input Pulse", color="red", linestyle="-")
plt.plot(t * 1e9, np.abs(output_asymmetric), label="Output Pulse", color="blue", linestyle="-.")
plt.grid(which='minor')
plt.title("Pulse Evolution (Soliton Pulse) (Asymmetric)")
plt.xlabel("Time (ns)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.show()

# Input Pulse (Gaussian)
input_pulse = np.sqrt(p0) * np.exp(-((1 + 1j * C) / (2 * t0_2)) * t**2)

# Symmetric Method (Gaussian)
output_symmetric = np.fft.fftshift(np.fft.fft(input_pulse))
for _ in range(num_of_steps):
    output_symmetric = d_operator(output_symmetric, beta2, dz / 2, freq)
    output_symmetric = np.fft.ifft(np.fft.ifftshift(output_symmetric))
    output_symmetric = n_operator(output_symmetric, gamma, dz)
    output_symmetric = np.fft.fftshift(np.fft.fft(output_symmetric))
    output_symmetric = d_operator(output_symmetric, beta2, dz / 2, freq)
output_symmetric = np.fft.ifft(np.fft.ifftshift(output_symmetric))

# Asymmetric Method (Gaussian)
output_asymmetric = np.fft.fftshift(np.fft.fft(input_pulse))
for _ in range(num_of_steps):
    output_asymmetric = d_operator(output_asymmetric, beta2, dz, freq)
    output_asymmetric = np.fft.ifft(np.fft.ifftshift(output_asymmetric))
    output_asymmetric = n_operator(output_asymmetric, gamma, dz)
    output_asymmetric = np.fft.fftshift(np.fft.fft(output_asymmetric))
output_asymmetric = np.fft.ifft(np.fft.ifftshift(output_asymmetric))

# Plotting Gaussian Pulse Evolution
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(t * 1e9, np.abs(input_pulse), label="Input Pulse", color="red", linestyle="-")
plt.plot(t * 1e9, np.abs(output_symmetric), label="Output Pulse", color="blue", linestyle="-.")
plt.grid(which='minor')
plt.title("Pulse Evolution (Gaussian Pulse) (Symmetric)")
plt.xlabel("Time (ns)")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t * 1e9, np.abs(input_pulse), label="Input Pulse", color="red", linestyle="-")
plt.plot(t * 1e9, np.abs(output_asymmetric), label="Output Pulse", color="blue", linestyle="-.")
plt.grid(which='minor')
plt.title("Pulse Evolution (Gaussian Pulse) (Asymmetric)")
plt.xlabel("Time (ns)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.show()
