import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from numpy.fft import fft, fftfreq


t = np.linspace(0, 10, 500)
low_freq = np.sin(2*np.pi*5*t)           # low frequency
high_freq_small = 0.5*np.sin(2*np.pi*15*t) # high frequency, low amplitude
combined = low_freq + high_freq_small

plt.figure(figsize=(10,4))
plt.plot(t, low_freq, label="Low frequency")
plt.plot(t, high_freq_small, label="High frequency (small amplitude)")
plt.plot(t, combined, label="Combined")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Low-frequency to high-frequency wave")
plt.show()


signal = combined
N = len(signal)
dt = t[1]-t[0]
freqs = fftfreq(N, dt)
spectrum = np.abs(fft(signal))**2

plt.figure(figsize=(10,4))
plt.plot(freqs[:N//2], spectrum[:N//2])
plt.xlabel("Frequency")
plt.ylabel("Energy")
plt.title("Spectral Energy Distribution")
plt.show()



#f, tt, Sxx = spectrogram(combined, fs=1/dt)

# Time axis
t = np.linspace(0, 10, 2000)
T = t[-1]   # total simulation time
tau = t/T   # normalized 0 → 1

# Transfer function (linear for now)
alpha = tau   # try sigmoid for smoother transfer
sigmoid = 1/(1 + np.exp(0.5*(tau-0.5)))

# Define signal with energy transfer: freq 5 → freq 15
f1, f2 = 5, 20
signal = (1-alpha)*np.sin(2*np.pi*f1*t) + 0.5*alpha*np.sin(2*np.pi*f2*t)


# Plot time-domain signal
plt.figure(figsize=(10,4))
plt.plot(t, signal)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Energy Transfer: 5 Hz → 15 Hz")
plt.show()

N = len(signal)
dt = t[1]-t[0]
freqs = fftfreq(N, dt)
spectrum = np.abs(fft(signal))**2
plt.figure(figsize=(10,4))
plt.plot(freqs[:N//2], spectrum[:N//2])
plt.xlabel("Frequency")
plt.ylabel("Energy")
plt.title("Spectral Energy Distribution")
plt.show()


# Plot spectrogram
f, tt, Sxx = spectrogram(signal, fs=200)  # fs = sampling freq
plt.figure(figsize=(6,6))
plt.pcolormesh(f ,tt, 10*np.log10(Sxx).T, shading='gouraud')
plt.xlim(0, 25)
plt.ylabel("Time (s)")
plt.xlabel("Frequency (Hz)")
plt.title("Spectrogram: Energy transfer from 5 Hz to 15 Hz")
plt.colorbar(label="Power (dB)")
plt.show()


