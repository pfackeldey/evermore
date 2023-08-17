import equinox as eqx

from dilax.likelihood import NLL, Hessian, CovMatrix

from examples.model import model, init_values, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)

# create negative log likelihood
nll = NLL(model=model, observation=observation)

# fit
params, state = optimizer.fit(fun=nll, init_values=init_values)

# gradients of nll of fitted model
fast_grad_nll = eqx.filter_jit(eqx.filter_grad(nll))
grads = fast_grad_nll(params)
# gradients of nll of fitted model only wrt to `mu`
# basically: pass the parameters dict of which you want the gradients
params_ = {k: v for k, v in params.items() if k == "mu"}
grad_mu = fast_grad_nll(params_)

# hessian + cov_matrix of nll of fitted model
hessian = eqx.filter_jit(Hessian(model=model, observation=observation))()

# covariance matrix of fitted model
covmatrix = eqx.filter_jit(CovMatrix(model=model, observation=observation))()
