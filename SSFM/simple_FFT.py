# how to transform a sine signal into fourier space
# we take a sine wave that contains 1 Hz , 5 Hz and 8 Hz
# we get similar results from FFT out put show spike on 1 Hz , 5 and 8 hz



import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

N =100
t = np.linspace(0, 5, N)
dt = t[1] - t[0]
y0 = np.sin(2*np.pi*t) # sin(2 pi f t) f is the frequency 1
y1 =  np.sin(2*np.pi*5*t) # frequency 5
y2 =  np.sin(2*np.pi*8*t) # frequency 8
y = y0 + y1 + y2
plt.plot(t, y)
plt.xlabel("time")
plt.ylabel("y value")
plt.show()



yf = fft(y)/N
k = fftfreq(N, dt)
plt.plot( k[:N//2], np.abs(yf)[:N//2])
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.show()












# below code uses rfft which is for real input 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq

N =100
t = np.linspace(0, 5, N)
dt = t[1] - t[0]
y0 = np.sin(2*np.pi*t) # sin(2 pi f t) f is the frequency 1
y1 =  np.sin(2*np.pi*5*t) # frequency 5
y2 =  np.sin(2*np.pi*8*t) # frequency 8
y = y0 + y1 + y2
plt.plot(t, y)
plt.xlabel("time")
plt.ylabel("y value")
plt.show()



yf = rfft(y)/N
k = rfftfreq(N, dt)
plt.plot( k, np.abs(yf))
plt.xlabel("frequency")
plt.ylabel("amplitude")
plt.show()




