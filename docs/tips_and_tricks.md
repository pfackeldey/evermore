---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Tips and Tricks

Here are some advanced tips and tricks.


## penzai Visualization

Use `penzai` to visualize evermore components!

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import evermore as evm
import equinox as eqx

jax.config.update("jax_enable_x64", True)


mu = evm.Parameter(value=1.1)
sigma1 = evm.NormalParameter(value=0.1)
sigma2 = evm.NormalParameter(value=0.2)

hist = jnp.array([10, 20, 30])


mu_mod = mu.scale(offset=0, slope=1)
sigma1_mod = sigma1.scale_log(up=1.1, down=0.9)
sigma2_mod = sigma2.scale_log(up=1.05, down=0.95)
composition = evm.modifier.Compose(
    mu_mod,
    sigma1_mod,
    evm.modifier.Where(hist < 15, sigma2_mod, sigma1_mod),
)
composition = evm.modifier.Compose(
    composition,
    evm.Modifier(parameter=sigma1, effect=evm.effect.AsymmetricExponential(up=1.2, down=0.8)),
)

evm.visualization.display(composition)
```



## Parameter Partitioning

For optimization it is necessary to differentiate only against meaningful leaves of the PyTree of `evm.Parameters`.
By default JAX would differentiate w.r.t. every non-static leaf of a `evm.Parameter`, including the prior PDF and its bounds.
Gradients are typically only wanted w.r.t. the `.value` attribute of the `evm.Parameters`. This is solved by splitting
the PyTree of `evm.Parameters` into a differentiable and a non-differentiable part, and then defining the loss function
w.r.t. both parts. Gradient calculation is performed only w.r.t. the differentiable arguement, see:

```{code-block} python
from jaxtyping import Array, PyTree
import evermore as evm

# define a PyTree of parameters
params = {
    "a": evm.Parameter(),
    "b": evm.Parameter(),
}

# split the PyTree into diffable and the static parts
filter_spec = evm.parameter.value_filter_spec(params)
diffable, static = eqx.partition(params, filter_spec)

# or
# diffable, static = evm.parameter.partition(params)

# loss's first argument is only the diffable part of the parameter Pytree!
def loss(diffable: PyTree[evm.Parameter], static: PyTree[evm.Parameter], hists: PyTree[Array]) -> Array:
    # combine the diffable and static parts of the parameter PyTree
    parameters = eqx.combine(diffable, static)
    assert parameters == params
    # use the parameters to calculate the loss as usual
    ...

grad_loss = eqx.filter_grad(loss)(diffable, static, ...)
```

If you need to further exclude parameter from being optimized you can either set `frozen=True` or set the corresponding leaf in `filter_spec` from `True` to `False`.


## JAX Transformations

Evert component of evermore is compatible with JAX transformations. That means you can `jax.jit`, `jax.vmap`, ... _everything_.
You can e.g. sample the parameter values multiple times vectorized from its prior PDF:

```{code-cell} ipython3
import jax
import evermore as evm


params = {"a": evm.NormalParameter(), "b": evm.NormalParameter()}

rng_key = jax.random.key(0)
rng_keys = jax.random.split(rng_key, 100)

vec_sample = jax.vmap(evm.parameter.sample, in_axes=(None, 0))

evm.visualization.display(vec_sample(params, rng_keys))
```

Many minimizers from the JAX ecosystem are e.g. batchable (`optax`, `optimistix`), which allows you vectorize _full fits_, e.g., for embarrassingly parallel likleihood profiles.

## Visualize the Computational Graph

You can visualize the computational graph of a JAX computation by:

```{code-block} python
import pathlib
import jax.numpy as jnp
import equinox as eqx
import evermore as evm

param = evm.Parameter(value=1.1)

# create the modifier and JIT it
modify = eqx.filter_jit(param.scale())

# apply the modifier
hist = jnp.array([10, 20, 30])
modify(hist)
# -> Array([11., 22., 33.], dtype=float32, weak_type=True),


# visualize the graph:
filepath = pathlib.Path('graph.gv')
filepath.write_text(evm.util.dump_hlo_graph(modify, hist), encoding='ascii')
```
