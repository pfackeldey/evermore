# Getting Started

evermore is a toolbox that provides common building blocks for building (binned)
likelihoods in high-energy physics with JAX.

## Binned Likelihood

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
