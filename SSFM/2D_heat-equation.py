# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 00:41:49 2023

@author: gauta
"""


# import 
import numpy as np
import matplotlib.pyplot as plt



"""
2 dimensional heat equation
 
temperature in disk


Finite difference
 forward difference in time
 central difference in space
 

du/dt = D(du2/dx2 + du2/dy2) 


Where, D is Thermal diffusivity


"""



# dimension of 2D plate
w = 10.   # plate size x axis
h = 10    # plate size y axis
D = 4 # Thermal diffusivity




dx = 0.1 # intervals in x- directions
dy = 0.1 # intervals in y- directions


dx2=dx**2
dy2=dy**2


# Temperature of plate at cold point and hot point
T_cool = 100     # temperature at cold point
T_hot = 700      # temperature at hot point


# No of points in x axis and y axis
Nx = int(w/dx)
Ny = int(h/dy)


# time element
dt = ((0.5/D)*(dx**2*dy**2))/(dx**2 + dy**2)


# initialization of U 

u0 = T_cool*np.zeros((Nx,Ny))   # initial condition
u = u0.copy()

# Initial conditions - circle of radius r centred at (cx,cy) (mm)
r = 2
cx = 6
cy = 5

for i in range(Nx):
    for j in range(Ny):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        if p2 < r**2:
            u0[i,j] = T_hot

# following function
#do_timestep updates the numpy array u from the results of the previous timestep, u0

def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx**2
                                + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy**2)

    u0 = u.copy()
    return u0, u

# Number of timesteps
n = 101



# propagation of temperature in the disk
# Output 4 figures at these timesteps
M = [0, 10, 50,100]
fignum = 0
fig = plt.figure()

for m in range(n):
    u0, u = do_timestep(u0, u)
    if m in M:
        fignum += 1
        #print(m, fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(u.copy(), cmap=plt.get_cmap('coolwarm'), vmin=T_cool, vmax=T_hot)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format(m*dt*1000))

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.savefig('2Dheat.png')
plt.show()
















