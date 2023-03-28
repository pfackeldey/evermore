import jax.numpy as jnp

from dilax.likelihood import nll, sample

from examples.model import model, init_params, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)

# fit
params, state = optimizer.fit(
    fun=nll, init_params=init_params, model=model, observation=observation
)

# sample toys based on fitted parameters
toys_postfit = sample(parameters=params, model=model, observation=observation, toys=100)
# sample toys based on initial parameters (before fit)
toys_prefit = sample(parameters=init_params, model=model, observation=observation, toys=100)
# calculate uncertainty from toys (postfit)
unc_postfit = jnp.std(toys_postfit, axis=0)
