import numpy as np
import torch

# differentiation of u wrt x


def dudx(u):
    deriv = 1/12*u[:,:,0] -2/3*u[:,:,1] + 0*u[:,:,2] + 2/3*u[:,:,3] - 1/12*u[:,:,4]
    return torch.t(derv).unsqueeze(0)

def d2udx2(u):
    deriv = -1/12*u[:,:,0] + 4/3*u[:,:,1] - 5/2*u[:,:,2] + 4/3*u[:,:,3] - 1/12*u[:,:,4]
    return torch.t(derv).unsqueeze(0)

def d3udx3(u):
    deriv = -1/2*u[:,:,0] + u[:,:,1] + 0*u[:,:,2] - 1*u[:,:,3] + 1/2*u[:,:,4]
    return torch.t(derv).unsqueeze(0)

def d4udx4(u):
    deriv = 1*u[:,:,0] - 4*u[:,:,1] + 6*u[:,:,2] - 4*u[:,:,3] + 1*u[:,:,4]
    return torch.t(derv).unsqueeze(0)
