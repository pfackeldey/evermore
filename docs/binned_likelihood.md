# Binned Likelihood

The binned likelihood quantifies the agreement between a model and data in terms
of histograms. It is defined as follows:

```{math}
:label: likelihood
\mathcal{L}(d|\phi) = \prod_{i}^{n} \frac{\lambda_i(\phi)^{d_i}}{d_i!} e^{-\lambda_i(\phi)} \cdot \prod_j^p \pi_j\left(\phi_j\right)
```

where {math}`\lambda_i(\phi)` is the model prediction for bin {math}`i`,
{math}`d_i` is the observed data in bin {math}`i`, and
{math}`\pi_j\left(\phi_j\right)` is the prior probability density function (PDF)
for parameter {math}`j`. The first product is a Poisson per bin, and the second
product is the constraint from each prior PDF.

Key to constructing this likelihood is the definition of the model
{math}`\lambda(\phi)` as a function of parameters {math}`\phi`. evermore
provides building blocks to define these in a modular way.

These building blocks include:

- **evm.Parameter**: A class that represents a parameter with a value, name,
  bounds, and prior PDF used as constraint.
- **evm.Effect**: Effects describe how data, e.g., histogram bins, may be
  varied.
- **evm.Modifier**: Modifiers combine **evm.Effects** and **evm.Parameters** to
  modify data.

The negative log-likelihood (NLL) function of Eq.{eq}`likelihood` can be implemented with evermore as follows (copy & paste the following snippet to start write a _new_ statistical model):

```{code-block} python
from jaxtyping import PyTree, Array
import equinox as eqx
import evermore as evm


# -- parameter definition --
# params: PyTree[evm.Parameter] = ...
# dynamic_params, static_params = evm.parameter.partition(params)


# -- model definition --
# def model(params: PyTree[evm.Parameter], hists: PyTree[Array]) -> PyTree[Array]:
#   ...


# -- NLL definition --
@eqx.filter_jit
def NLL(dynamic_params, static_params, hists, observation):
    params = eqx.combine(dynamic_params, static_params)
    expectations = model(params, hists)

    # first product of Eq. 1 (Poisson term)
    loss_val = evm.pdf.Poisson(lamb=evm.util.sum_over_leaves(expectations)).log_prob(
        observation
    ).sum()

    # second product of Eq. 1 (constraint)
    constraints = evm.loss.get_log_probs(model)
    # for parameters with `.value.size > 1` (jnp.sum the constraints)
    constraints = jax.tree.map(jnp.sum, constraints)
    loss_val += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(loss_val)
```

Building the parameters and the model is key here. The relevant parts to build parameters and a model are described in <project:#building-blocks>.
