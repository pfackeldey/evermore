import jax
import jax.numpy as jnp

from dilax.likelihood import nll

from examples.model import model, init_params, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)


# fit
params, state = optimizer.fit(
    fun=nll, init_params=init_params, model=model, observation=observation
)

# gradients of nll of fitted model
grads = jax.grad(nll, argnums=0)(params, model, observation)
# gradients of nll of fitted model only wrt to `mu`
# basically: pass the parameters dict of which you want the gradients
params_ = {k: v for k, v in params.items() if k == "mu"}
grad_mu = jax.grad(nll, argnums=0)(params_, model, observation)

# hessian + cov_matrix of nll of fitted model
hessian = jax.hessian(nll, argnums=0)(params, model, observation)
hessian, _ = jax.tree_util.tree_flatten(hessian)
hessian = jnp.reshape(jnp.array(hessian), (len(params), len(params)))

# covariance matrix of fitted model
cov_matrix = jnp.linalg.inv(hessian)

# or simply
from dilax.likelihood import cov_matrix

covmatrix = cov_matrix(params, model, observation)
