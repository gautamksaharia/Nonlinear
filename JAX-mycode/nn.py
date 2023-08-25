def init_params(layers):
  keys = jax.random.split(jax.random.PRNGKey(0),len(layers)-1)
  params = list()
  for key,n_in,n_out in zip(keys,layers[:-1],layers[1:]):
    lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in)) # xavier initialization lower and upper bound
    W = lb + (ub-lb) * jax.random.uniform(key,shape=(n_in,n_out))
    B = jax.random.uniform(key,shape=(n_out,))
    params.append({'W':W,'B':B})
  return params

def fwd(params,t):
  X = jnp.concatenate([t],axis=1)
  *hidden,last = params
  for layer in hidden :
    X = jax.nn.tanh(X@layer['W']+layer['B'])
  return X@last['W'] + last['B']

@jax.jit
def MSE(true,pred):
  return jnp.mean((true-pred)**2)

def loss_fun(params,colloc,conds):
  t_c =colloc[:,[0]]
  ufunc = lambda t : fwd(params,t)
  ufunc_t=lambda t:jax.grad(lambda t:jnp.sum(ufunc(t)))(t)
  loss =jnp.mean(ODE_loss(t_c,ufunc) **2)

  t_ic,u_ic = conds[0][:,[0]],conds[0][:,[1]]  
  loss += MSE(u_ic,ufunc(t_ic))

  t_bc,u_bc = conds[1][:,[0]],conds[1][:,[1]]  
  loss += MSE(u_bc,ufunc_t(t_bc))

  return  loss

@jax.jit
def update(opt_state,params,colloc,conds):
  # Get the gradient w.r.t to MLP params
  grads=jax.jit(jax.grad(loss_fun,0))(params,colloc,conds)
  
  #Update params
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  #Update params
  # return jax.tree_multimap(lambda params,grads : params-LR*grads, params,grads)
  return opt_state,params
