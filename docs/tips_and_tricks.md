# Tips and Tricks

Here are some advanced tips and tricks.


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

```{code-block} python
import jax
import evermore as evm


params = {"a": evm.NormalParameter(), "b": evm.NormalParameter()}

rng_key = jax.random.key(0)
rng_keys = jax.random.split(rng_key, 100)

vec_sample = jax.vmap(evm.parameter.sample, in_axes=(None, 0))

print(vec_sample(params, rng_keys))
# {'a': NormalParameter(
#   value=f32[100,1],
#   name=None,
#   lower=f32[100,1],
#   upper=f32[100,1],
#   prior=Normal(mean=f32[100,1], width=f32[100,1]),
#   frozen=False,
# ),
#  'b': NormalParameter(
#   value=f32[100,1],
#   name=None,
#   lower=f32[100,1],
#   upper=f32[100,1],
#   prior=Normal(mean=f32[100,1], width=f32[100,1]),
#   frozen=False,
# )}
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
