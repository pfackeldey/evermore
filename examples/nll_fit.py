from dilax.likelihood import NLL

from examples.model import model, init_values, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)


# create negative log likelihood
nll = NLL(model=model, observation=observation)

# fit
values, state = optimizer.fit(fun=nll, init_values=init_values)

# update model with fitted values
fitted_model = model.update(values=values)
