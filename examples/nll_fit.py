from dilax.likelihood import nll

from examples.model import model, init_params, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)


# fit
params, state = optimizer.fit(
    fun=nll, init_params=init_params, model=model, observation=observation
)

# update model with fitted parameters
fitted_model = model.apply(parameters=params)
