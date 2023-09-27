from __future__ import annotations

from jax.config import config
from model import init_values, model, observation, optimizer

from dilax.likelihood import NLL

config.update("jax_enable_x64", True)


# create negative log likelihood
nll = NLL(model=model, observation=observation)

# fit
values, state = optimizer.fit(fun=nll, init_values=init_values)

# update model with fitted values
fitted_model = model.update(values=values)
