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

evermore components can be visualized with [treescope](https://treescope.readthedocs.io/en/stable/index.html). In IPython notebooks you can display the tree using `treescope.display`.

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import evermore as evm
import equinox as eqx
import treescope

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

with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
    treescope.display(composition)
```

You can also save the tree to an HTML file.
```{code-cell} python
with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
    contents = treescope.render_to_html(composition)

with open("composition.html", "w") as f:
    f.write(contents)
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

# unwrap (or "transform")
pytree_t = evm.transform.unwrap(pytree)
# wrap back (or "inverse transform")
pytree_tt = evm.transform.wrap(pytree_t)

wl.pprint(("Original", pytree), width=150, short_arrays=False)
wl.pprint(("Transformed", pytree_t), width=150, short_arrays=False)
wl.pprint(("Transformed back / Original", pytree_tt), width=150, short_arrays=False)
```

Transformations always transform into the unconstrained real space (using [`evm.transform.unwrap`](#evermore.parameters.transform.unwrap)) and back to the constrained space (using [`evm.transform.wrap`](#evermore.parameters.transform.wrap)).
Typically, you would transform your parameters as a first step inside your loss (or model) function.
Then, a minimizer can optimize the transformed parameters in the unconstrained space. Finally, you can transform them back to the constrained space for further processing.

Custom transformations can be defined by subclassing [`evm.transform.ParameterTransformation`](#evermore.parameters.transform.AbstractParameterTransformation) and implementing the [`wrap`](#evermore.parameters.transform.AbstractParameterTransformation.wrap) and [`unwrap`](#evermore.parameters.transform.AbstractParameterTransformation.unwrap) methods.


## Parameter Partitioning

For optimization it is necessary to differentiate only against meaningful leaves of the PyTree of `evm.Parameters`.
By default JAX would differentiate w.r.t. every non-static leaf of a `evm.Parameter`, including the prior PDF and its bounds.
Gradients are typically only wanted w.r.t. the `.value` attribute of the `evm.Parameters`. This is solved by splitting
the PyTree of `evm.Parameters` into a differentiable and a non-differentiable part, and then defining the loss function
w.r.t. both parts. Gradient calculation is performed only w.r.t. the differentiable argument, see:

```{code-block} python
from jaxtyping import Array, PyTree
import equinox as eqx
import evermore as evm

# define a PyTree of parameters
params = {
    "a": evm.Parameter(),
    "b": evm.Parameter(),
}

dynamic, static = evm.tree.partition(params)
print(f"{dynamic=}")
print(f"{static=}")

# loss's first argument is only the dynamic part of the parameter Pytree!
def loss(dynamic: PyTree[evm.Parameter], static: PyTree[evm.Parameter], hists: PyTree[Array]) -> Array:
    # combine the dynamic and static parts of the parameter PyTree
    parameters = evm.tree.combine(dynamic, static)
    assert parameters == params
    # use the parameters to calculate the loss as usual
    ...

grad_loss = eqx.filter_grad(loss)(dynamic, static, ...)
```

If you need to further exclude parameter from being optimized you can either set `frozen=True`.
For a more precise handling of the partitioning and combining step, have a look at `eqx.partition`, `eqx.combine`, and `evm.tree.value_filter_spec`.


(tree-manipulations)=
## PyTree Manipulations

`evermore` provides (similarly to `nnx`) a little filter DSL to allow more powerful manipulations of PyTrees of `evm.Parameters`.
The following example highlights the `evm.tree.only` function using various filters:

```{code-cell} ipython3
import evermore as evm
import wadler_lindig as wl

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

# parameter-only pytree
params = evm.tree.only(tree, evm.filter.is_parameter)
print("evm.filter.is_parameter:")
wl.pprint(params, width=200)

print("\nevm.filter.is_frozen:")
wl.pprint(evm.tree.only(params, evm.filter.is_frozen), width=200)

print("\nevm.filter.is_not_frozen:")
wl.pprint(evm.tree.only(params, evm.filter.is_not_frozen), width=200)

print("\nevm.filter.HasName('mu'):")
wl.pprint(evm.tree.only(params, evm.filter.HasName("mu")), width=200)

print("\nevm.filter.HasTags({'theory'}):")
wl.pprint(evm.tree.only(params, evm.filter.HasTags(tags)), width=200)
```

`evm.tree.partition` also accepts a `filter` argument, and let's you partition any pytree as you want.


## JAX Transformations

Evert component of evermore is compatible with JAX transformations. That means you can `jax.jit`, `jax.vmap`, ... _everything_.
You can e.g. sample the parameter values multiple times vectorized from its prior PDF:

```{code-cell} ipython3
import jax
import evermore as evm
import treescope


params = {"a": evm.NormalParameter(), "b": evm.NormalParameter()}

rng_key = jax.random.key(0)
rng_keys = jax.random.split(rng_key, 10_000)

vec_sample = jax.vmap(evm.sample.sample_from_priors, in_axes=(None, 0))

with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
    tree = vec_sample(params, rng_keys)
    treescope.display(tree)
```

Many minimizers from the JAX ecosystem are e.g. batchable (`optax`, `optimistix`), which allows you vectorize _full fits_, e.g., for embarrassingly parallel likelihood profiles.

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
