import jax 
import jax.numpy as jnp

# u_t + u_tt = t*cos(2*pi*t)
def ODE_loss(t,u):
  """
  t: float 
  t is time axis
  u : function u(t)
   
  ODE loss is residual function of the ODE
  u_t + u_tt = t*cos(2*pi*t)
  """
  u_t=lambda t:jax.grad(lambda t:jnp.sum(u(t)))(t)
  u_tt=lambda t:jax.grad(lambda t : jnp.sum(u_t(t)))(t)
  return -t*jnp.cos(2*jnp.pi*t) + u_t(t) + u_tt(t)
