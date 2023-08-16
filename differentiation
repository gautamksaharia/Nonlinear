"""
Created on Wed Aug 16 12:35:46 2023

@author: gautamksaharia
"""


import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt




def D1(y, x):
    """ Differentiation of y wrt x using Automatic differentiation of PyTorch
    dy/dx
    x : variable . must be requires_grad=True
    """
    dydx = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
    return dydx


def D2(y, x):
    """ Differentiation of y wrt x using Automatic differentiation of PyTorch
    dy2/dx2
    """
    dydx = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
    dy2dx2 = grad(dydx.sum(), x, retain_graph=True, create_graph=True)[0]
    return dy2dx2


def D3(y, x):
    """ Differentiation of y wrt x using Automatic differentiation of PyTorch
    dy3/dx3
    """
    dydx = grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
    dy2dx2 = grad(dydx.sum(), x, retain_graph=True, create_graph=True)[0]
    dy3dx3 = grad(dy2dx2.sum(), x, retain_graph=True, create_graph=True)[0]
    return dy3dx3



def dudx(u, dx):
    """
    Approximate the first derivative using the centered finite difference
    formula.
    dx : float space element
    """
    first_deriv = np.zeros_like(u)

    # wrap to compute derivative at endpoints
    first_deriv[0] = (u[1] - u[-1]) / (2*dx)
    first_deriv[-1] = (u[0] - u[-2]) / (2*dx)

    # compute du/dx for all the other points
    first_deriv[1:-1] = (u[2:] - u[0:-2]) / (2*dx)

    return first_deriv


def d2udx2(u, dx):
    """
    Approximate the second derivative using the centered finite difference
    formula.
    """
    second_deriv = np.zeros_like(u)

    # wrap to compute second derivative at endpoints
    second_deriv[0] = (u[1] - 2*u[0] + u[-1]) / (dx**2)
    second_deriv[-1] = (u[0] - 2*u[-1] + u[-2]) / (dx**2)

    # compute d2u/dx2 for all the other points
    second_deriv[1:-1] = (u[2:] - 2*u[1:-1] + u[0:-2]) / (dx**2)

    return second_deriv
