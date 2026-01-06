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

(treescope-visualization)=
## treescope Visualization

evermore components can be visualized with [treescope](https://treescope.readthedocs.io/en/stable/index.html). In IPython notebooks you can display the tree using `nnx.display`.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import evermore as evm
from flax import nnx
import treescope

jax.config.update("jax_enable_x64", True)


mu = evm.Parameter(value=1.1)
sigma1 = evm.NormalParameter(value=0.1)
sigma2 = evm.NormalParameter(value=0.2)

hist = jnp.array([10, 20, 30])


mu_mod = mu.scale(offset=0, slope=1)
sigma1_mod = sigma1.scale_log_asymmetric(up=1.1, down=0.9)
sigma2_mod = sigma2.scale_log_asymmetric(up=1.05, down=0.95)
composition = evm.modifier.Compose(
    mu_mod,
    sigma1_mod,
    evm.modifier.Where(hist < 15, sigma2_mod, sigma1_mod),
)
composition = evm.modifier.Compose(
    composition,
    evm.Modifier(value=sigma1.get_value(), effect=evm.effect.AsymmetricExponential(up=1.2, down=0.8)),
)

nnx.display(composition)
```

(parameter-transformations)=
## Parameter Transformations

evermore provides a set of parameter transformations that can be used to modify the parameter values.
This can be useful for example to ensure that the parameter values are within a certain range or that they are always positive.
evermore provides two predefined transformations: [`evm.transform.MinuitTransform`](#evermore.parameters.transform.MinuitTransform) (for bounded parameters) and [`evm.transform.SoftPlusTransform`](#evermore.parameters.transform.SoftPlusTransform) (for positive parameters).


```{code-cell} python
import evermore as evm
import wadler_lindig as wl


enforce_positivity = evm.transform.SoftPlusTransform()
pytree = {
    "a": evm.Parameter(2.0, transform=enforce_positivity),
    "b": evm.Parameter(0.1, transform=enforce_positivity),
}

nnx.display({"Original": pytree})

# unwrap (or "transform")
pytree_t = evm.transform.unwrap(pytree)
nnx.display({"Transformed": pytree_t})

# wrap back (or "inverse transform")
pytree_tt = evm.transform.wrap(pytree_t)
nnx.display({"Transformed back / Original": pytree_tt})
```

Transformations always transform into the unconstrained real space (using [`evm.transform.unwrap`](#evermore.parameters.transform.unwrap)) and back to the constrained space (using [`evm.transform.wrap`](#evermore.parameters.transform.wrap)).
Typically, you would transform your parameters as a first step inside your loss (or model) function.
Then, a minimizer can optimize the transformed parameters in the unconstrained space. Finally, you can transform them back to the constrained space for further processing.

Custom transformations can be defined by subclassing [`evm.transform.ParameterTransformation`](#evermore.parameters.transform.BaseParameterTransformation) and implementing the [`wrap`](#evermore.parameters.transform.BaseParameterTransformation.wrap) and [`unwrap`](#evermore.parameters.transform.BaseParameterTransformation.unwrap) methods.


## Parameter Partitioning

For optimization it is necessary to differentiate only against meaningful leaves of the PyTree of `evm.Parameters`.
By default JAX would differentiate w.r.t. every non-static leaf of a `evm.Parameter`, including the prior PDF and its bounds.
Gradients are typically only wanted w.r.t. the `.value` attribute of the `evm.Parameters`. This is solved by splitting
the PyTree of `evm.Parameters` into a differentiable and a non-differentiable part, and then defining the loss function
w.r.t. both parts. Gradient calculation is performed only w.r.t. the differentiable argument, see:

```{code-block} python
from flax import nnx
from jaxtyping import Array, PyTree
import evermore as evm

# define a PyTree of parameters
params = {
    "a": evm.Parameter(),
    "b": evm.Parameter(),
}

graphdef, dynamic, static = nnx.split(
    params, evm.filter.is_dynamic_parameter, ...
)
print(f"{nnx.pure(dynamic)=}")
print(f"{nnx.pure(static)=}")


# loss's first argument is only the dynamic part of the parameter PyTree!
def loss(dynamic_state: PyTree[evm.Parameter], args) -> Array:
    graphdef, static_state, hists = args
    # combine the dynamic and static parts of the parameter PyTree
    parameters = nnx.merge(graphdef, dynamic_state, static_state)
    assert parameters == params
    # use the parameters to calculate the loss as usual
    ...

    hists: PyTree[Array] = ...
args = (graphdef, static, hists)
grad_loss = nnx.grad(loss)(dynamic, args)
```

If you need to further exclude parameter from being optimized you can either set `frozen=True`.
For a more precise handling of the partitioning and combining step, have a look at `nnx.split`, `nnx.merge`, and `nnx.state`.


(tree-manipulations)=
## PyTree Manipulations

`evermore` provides (similarly to `nnx`) a little filter DSL to allow more powerful manipulations of PyTrees of `evm.Parameters`.
The following example highlights using `nnx.state(...).filter(...)` with various filters:

```{code-cell} ipython3
from flax import nnx
import evermore as evm

tags = frozenset({"theory"})

# some pytree of parameters and something else
tree = {
    "mu": evm.Parameter(name="mu"),
    "xsecs": {
        "dy": evm.Parameter(tags=tags),
        "tt": evm.Parameter(frozen=True, tags=tags),
    },
    "not_a_parameter": 0.0,
}

params_state, _ = nnx.state(tree, evm.filter.is_parameter, ...)
print("evm.filter.is_parameter:")
nnx.display(nnx.pure(params_state))

print("\nevm.filter.is_frozen:")
nnx.display(nnx.pure(params_state.filter(evm.filter.is_frozen)))

print("\nevm.filter.is_not_frozen:")
nnx.display(nnx.pure(params_state.filter(evm.filter.is_not_frozen)))

print("\nevm.filter.HasName('mu'):")
nnx.display(nnx.pure(params_state.filter(evm.filter.HasName("mu"))))

print("\nevm.filter.HasTags({'theory'}):")
nnx.display(nnx.pure(params_state.filter(evm.filter.HasTags(tags))))
```

`nnx.split` also accepts a `filter` argument, and lets you partition any PyTree as you want.


## JAX Transformations

Evert component of evermore is compatible with JAX transformations. That means you can `jax.jit`, `jax.vmap`, ... _everything_.
You can e.g. sample the parameter values multiple times vectorized from its prior PDF:

```{code-cell} ipython3
import evermore as evm
from flax import nnx
from functools import partial


params = {"a": evm.NormalParameter(), "b": evm.NormalParameter()}
rngs = nnx.Rngs(0)

# Single sample
sample = evm.sample.sample_from_priors(rngs, params)
nnx.display(sample)

# Batched sampling with independent RNG streams (total: 10_000 samples)
@nnx.split_rngs(splits=10_000)
@partial(nnx.vmap, in_axes=(0, None))
def batched_sample_from_priors(rngs, params):
    return evm.sample.sample_from_priors(rngs, params)

nnx.display(batched_sample_from_priors(rngs, params))
```

Many minimizers from the JAX ecosystem are e.g. batchable (`optax`, `optimistix`), which allows you vectorize _full fits_, e.g., for embarrassingly parallel likelihood profiles.

## Visualize the Computational Graph

You can visualize the computational graph of a JAX computation by:

```{code-block} python
import pathlib
import jax.numpy as jnp
from flax import nnx
import evermore as evm

param = evm.Parameter(value=1.1)

# create the modifier and JIT it
modify = nnx.jit(param.scale())

# apply the modifier
hist = jnp.array([10, 20, 30])
modify(hist)
# -> Array([11., 22., 33.], dtype=float32, weak_type=True),


# visualize the graph:
filepath = pathlib.Path('graph.gv')
filepath.write_text(evm.util.dump_hlo_graph(modify, hist), encoding='ascii')
```
