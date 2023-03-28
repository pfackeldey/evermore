from dilax.pulls_impacts import profile_parameter

from examples.model import model, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)


# postfit nll profile of `norm2`; prefit nll profile with `fit=False`
nll_vals = profile_parameter(
    param_name="norm2",
    scan_points=model.parameters["norm2"].default_scan_points,
    model=model,
    observation=observation,
    fit=True,
    optimizer=optimizer,
)
