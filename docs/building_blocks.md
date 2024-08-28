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

(building-blocks)=
# Building Blocks

## Parameter

A parameter {math}`\phi` (see Eq.{eq}`likelihood`) is defined in evermore by `evm.Parameter`.
It holds a `value` which holds a single value or an array of values that can be optimized during a fit.
In the case of binned fits (working with histograms) a single floating point value may scale bins together, whereas
as an array of values may scale each bin independently (given that `value.shape == hist.shape`).

A parameter can be constructed as follows:

```{code-block} python
import evermore as evm


# simple parameter
parameter = evm.Parameter()

# options
parameter = evm.Parameter(
    value=1.0,          # default: 0.0
    name="my_param",    # default: None
    lower=0.0,          # default: -jnp.inf
    upper=10.0,         # default: +jnp.inf
    prior=None,         # default: None
    frozen=False,       # default: False
)
```

PDFs

:   In typical HEP analysis three prior constraint PDFs are most commonly used: `None`, `Normal`, `Poisson`.
    The latter one is typically only used for statistical uncertainties and for the first product of the binned likelihood (see Eq.{eq}`likelihood`).
    The first two are used most commonly within HEP analysis, thus evermore provides short-hands to create parameters with these PDFs:

    ```{code-block} python
    import evermore as evm


    # parameter with no constraint, `prior=None` (default)
    parameter = evm.Parameter()

    # parameter with standardized Normal constraint, `prior=Normal(mean=0, width=1)`
    parameter = evm.NormalParameter()

    # or explicit
    parameter = evm.Parameter(prior=evm.pdf.Normal(mean=0.0, width=1.0))
    ```

    :::{tip}

    You can use _any_ JAX-compatible PDF that satisfies the `evm.custom_types.PDFLike` protocol that requires a `.log_prob` and a `.sample` method to be present.
    Examples would be PDFs from `distrax` or from the JAX-substrate of TensorFlow Probability.


Parameter Boundaries

:   The `lower` and `upper` attributes denote the valid bounds of a parameter. Using `evm.loss.get_boundary_constraints` allows you to extract
    from a PyTree of parameters if a value if outside of these bounds; if yes it return `jnp.inf` else `0.0`. This return value can be added to
    the likelihood function in order to _break_ the fit in case a parameter runs out of its bounds.


Freeze a Parameter

:   For the minimization of a likelihood it is necessary to differentiate with respect to the _differentiable_ part, i.e., the `.value` attributes, of a PyTree of `evm.Parameters`.
    Splitting this tree into the differentiable and non-differentiable part is done with `evm.parameter.partition`. You can freeze a `evm.Parameter` by setting `frozen=True`, this will
    put the frozen parameter in the non-differentiable part.

Correlate a Parameter

:   Correlating a parameter is done by just using the same parameter instance for different modifiers. If this is - for whatever reason - not possible, evermore provides a helper to correlate parameters:
    ```{code-block} python
    from jaxtyping import PyTree
    import evermore as evm


    p1 = evm.Parameter(value=1.0)
    p2 = evm.Parameter(value=0.0)
    p3 = evm.Parameter(value=0.5)

    # correlate them
    p1, p2, p3 = evm.parameter.correlate(*parameters)

    # now p1, p2, p3 are correlated, i.e., they share the same value
    assert p1.value == p2.value == p3.value
    ```

    A more general case of correlating any PyTree of parameters is implemented as follows:
    ```{code-block} python
    from typing import NamedTuple


    class Params(NamedTuple):
        mu: evm.Parameter
        syst: evm.NormalParameter

    params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))

    flat_params, tree_def = jax.tree.flatten(params, evm.parameter.is_parameter)

    # correlate the parameters
    correlated_flat_params = evm.parameter.correlate(*flat_params)
    correlated_params = jax.tree.unflatten(tree_def, correlated_flat_params)

    # now correlated_params.mu and correlated_params.syst are correlated,
    # they share the same value
    assert correlated_params.mu.value == correlated_params.syst.value
    ```


:::{admonition} Inspect `evm.Parameters` with `treescope`
:class: tip dropdown

Inspect a (PyTree of) `evm.Parameters` with [treescope](https://treescope.readthedocs.io/en/stable/index.html) visualization in IPython or Colab notebooks (see <project:#treescope-visualization> for more information).
You can even add custom visualizers, such as:

```{code-block} python
import evermore as evm


tree = {"a": evm.NormalParameter(), "b": evm.NormalParameter()}

with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
    treescope.display(tree)
```
:::

## Effect

Effects describe how data ({math}`d`), i.e., histogram bins, are varied as a function of `evm.Parameters` ({math}`\phi`).
They return multiplicative ({math}`\alpha`) and additive ({math}`\Delta`) variations that are applied to the data as follows:

```{math}
:label: OffsetAndScale
\widetilde{d}\left(\phi\right) = \alpha\left(\phi\right) \cdot \left(d + \Delta\left(\phi\right) \right).
```

For binned likelihoods in HEP, evermore pre-defines the most common types of effects:


Linear Scaling

:   `evm.effect.Linear` allows to scale data based on a linear function with a `slope` and an `offset`.
    This effect returns multiplicative variation.


Vertical Template Morphing

:   `evm.effect.VerticalTemplateMorphing` scales histograms based on two reference histograms that correspond to the {math}`+1\sigma` and {math}`-1\sigma` template variations.
    The mathematical formula of the interpolation between the template variations is explained [here](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/what_combine_does/model_and_likelihood/#shape-morphing-effects).


Asymmetric Exponential Scaling

:   `evm.effect.AsymmetricExponential` scales data based on an `up` and a `down` value. Outside these values the data is scaled linearily, inside based on an exponential interpolation.
    The mathematical description can be found [here](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/what_combine_does/model_and_likelihood/#normalization-effects).


Custom effects can be either implemented by inheriting from `evm.effect.Effect` or - more conveniently - be defined with `evm.effect.Lambda`.
Exemplary, a custom effect that varies a 3-bin histogram by a constant absolute {math}`1\sigma` uncertainty of `[1.0, 1.5, 2.0]` and returns an additive (`normalize_by="offset"`) variation:

```{code-block} python
from jaxtyping import Array
import jax.numpy as jnp
import evermore as evm


def fun(parameter: evm.Parameter, hist: Array) -> Array:
    return hist + parameter.value * jnp.array([1.0, 1.5, 2.0])

custom_effect = evm.effect.Lambda(fun, normalize_by="offset")
```

:::{admonition} Multi-Parameter Custom Effects
:class: tip dropdown

Custom effects can accept multiple `evm.Parameters`, e.g., a PyTree of `evm.Parameters`:

```{code-block} python
from jaxtyping import Array, PyTree
import jax.numpy as jnp
import evermore as evm


def fun(parameter: PyTree[evm.Parameter], hist: Array) -> Array:
    return parameter["slope"].value * hist + parameter["intercept"].value * jnp.array([1.0, 1.5, 2.0])

custom_effect = evm.effect.Lambda(fun, normalize_by="offset")

# use with `evm.Modifier` as follows:
custom_modifier = evm.Modifier(
    parameter={
        "slope": evm.Parameter(),
        "intercept": evm.NormalParameter(),
    },
    effect=custom_effect,
)
```
:::


## Modifier

Modifiers combine `evm.Parameters` and `evm.effect.Effects` and can _apply_ the variation as defined in Eq.{eq}`OffsetAndScale` to a histogram.

```{code-block} python
import jax.numpy as jnp
import evermore as evm

param = evm.Parameter(value=1.1)

# create the modifier
modify = evm.Modifier(parameter=param, effect=evm.effect.Linear(offset=0, slope=1))

# apply the modifier
modify(jnp.array([10, 20, 30]))
# -> Array([11., 22., 33.], dtype=float32, weak_type=True),
```

For the most common types of modifiers evermore provides shorthands that construct a modifier directly from parameters, two examples:

Modifier that scales a histogram with its value (no constraint):
:   ```{code-block} python
    import jax.numpy as jnp
    import evermore as evm

    param = evm.Parameter(value=1.1)

    # create the modifier from the previous code block
    modify = param.scale()

    # apply the modifier
    modify(jnp.array([10, 20, 30]))
    # -> Array([11., 22., 33.], dtype=float32, weak_type=True),
    ```


Modifier that scales a histogram based on vertical template morphing (Normal constraint):
:   ```{code-block} python
    import jax.numpy as jnp
    import evermore as evm


    param = evm.NormalParameter(value=1.2)

    # create the modifier
    modify = param.morphing(
        up_template=[12, 23, 35],
        down_template=[9, 17, 26],
    )

    # apply the modifier
    modify(jnp.array([10, 20, 30]))
    # -> Array([10.336512, 20.6, 30.936512], dtype=float32)
    ```

Multiple modifiers should be combined using `evm.modifier.Compose` or the `@` operator:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import evermore as evm
import treescope


jax.config.update("jax_enable_x64", True)

param = evm.NormalParameter(value=0.1)

modifier1 = param.morphing(
    up_template=[12, 23, 35],
    down_template=[9, 17, 26],
)

modifier2 = param.scale_log(up=1.1, down=0.9)

# apply the composed modifier
(modifier1 @ modifier2)(jnp.array([10, 20, 30]))
# -> Array([10.259877, 20.500944, 30.760822], dtype=float32)

with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
    composition = modifier1 @ modifier2
    treescope.display(composition)
```
