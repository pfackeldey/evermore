from dilax.pulls_impacts import pulls

from examples.model import model, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)

# calculate pulls of all model parameters
parameter_pulls = pulls(
    poi="mu",
    model=model,
    observation=observation,
    parameters=None,
    optimizer=optimizer,
)
