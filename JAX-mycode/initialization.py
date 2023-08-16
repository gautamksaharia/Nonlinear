# import modules
import numpy as np
import jax.numpy as jnp

# Xavier initialization

"""
The goal of Xavier Initialization is to initialize the weights such that 
the variance of the activations are the same across every layer. 
This constant variance helps prevent the gradient from exploding or vanishing.

draw each weight, w, from a normal distribution with mean of 0, and a standard deviation 
sigma= Sqrt(2/(inputs+outputs))

"""
def xavier_init(in_dim, out_dim):
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    W = jnp.array(glorot_stddev * np.random.normal(size=(in_dim, out_dim)))
    b = jnp.zeros(out_dim)
    return W, b
