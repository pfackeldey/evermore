import jax
import jax.numpy as jnp
import equinox as eqx

from dilax.likelihood import NLL, SampleToy

from examples.model import model, init_values, observation, optimizer

from jax.config import config

config.update("jax_enable_x64", True)


# create negative log likelihood
nll = NLL(model=model, observation=observation)

# fit
values, state = optimizer.fit(fun=nll, init_values=init_values)

# create sampling method
sample_toy = SampleToy(model=model, observation=observation)
# vectorise and jit
sample_toys = eqx.filter_vmap(in_axes=(None, 0))(eqx.filter_jit(sample_toy))

sample_toy(values, jax.random.PRNGKey(1234))

# sample 10 toys based on fitted parameters
keys = jax.random.split(jax.random.PRNGKey(1234), num=10)
# postfit toys
toys_postfit = sample_toys(values, keys)
# prefit toys
toys_prefit = sample_toys(init_values, keys)
