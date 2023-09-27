from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.model import Model


class BaseModule(eqx.Module):
    """
    Base module to hold the `model` and the `observation`.
    """

    model: Model
    observation: jax.Array = eqx.field(converter=jnp.asarray)

    def __init__(self, model: Model, observation: jax.Array) -> None:
        self.model = model
        self.observation = observation


class NLL(BaseModule):
    """
    Negative log-likelihood (NLL).

    Example:
    ```
        from dilax.model import Model
        from dilax.parameter import r, lnN

        class MyModel(Model):
            def evaluate(self) -> EvaluationResult:
                expectations = {}
                # signal
                signal, mu_penalty = self.parameters["mu"](self.processes["signal"], type="r")
                expectations["signal"] = signal

                # background
                background, sigma_penalty = self.parameters["sigma"](self.processes["background"], type="lnN", width=1.1)
                expectations["background"] = background
                return EvaluationResult(expectations=expectations, penalty=mu_penalty + sigma_penalty)

        model = MyModel(
            processes={"signal": jnp.array([10]), "background": jnp.array([50])},
            parameters={"mu": Parameter(value=1.0, bounds=(0, 100)), "sigma": Parameter(value=0, bounds=(-100, 100))},
        )
        observation = jnp.array([60])

        nll = NLL(model=model, observation=observation)

        # evaluate the negative log likelihood
        %timeit nll(values={"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])})
        >> 2.03 ms ± 150 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

        # evaluate the negative log likelihood *fast*
        %timeit eqx.filter_jit(nll)(values={"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])}).block_until_ready()
        >> 274 µs ± 3.6 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    ```
    """

    def logpdf(self, *args, **kwargs) -> jax.Array:
        return jax.scipy.stats.poisson.logpmf(*args, **kwargs)

    def __call__(self, values: dict[str, jax.Array] = {}) -> jax.Array:
        model = self.model.update(values=values)
        res = model.evaluate()
        nll = (
            self.logpdf(self.observation, res.expectation())
            - self.logpdf(self.observation, self.observation)
            + model.nll_boundary_penalty()
            + model.parameter_constraints()
        )
        return -jnp.sum(nll, axis=-1)


class Hessian(BaseModule):
    """
    Covariance Matrix.

    Example:
    ```
        from dilax.model import Model
        from dilax.parameter import r, lnN

        class MyModel(Model):
            def evaluate(self) -> EvaluationResult:
                expectations = {}
                # signal
                signal, mu_penalty = self.parameters["mu"](self.processes["signal"], type="r")
                expectations["signal"] = signal

                # background
                background, sigma_penalty = self.parameters["sigma"](self.processes["background"], type="lnN", width=1.1)
                expectations["background"] = background
                return EvaluationResult(expectations=expectations, penalty=mu_penalty + sigma_penalty)

        model = MyModel(
            processes={"signal": jnp.array([10]), "background": jnp.array([50])},
            parameters={"mu": Parameter(value=1.0, bounds=(0, 100)), "sigma": Parameter(value=0, bounds=(-100, 100))},
        )
        observation = jnp.array([60])

        hessian = Hessian(model=model, observation=observation)

        # evaluate the negative log likelihood
        %timeit hessian(values={"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])})
        >> 18.8 ms ± 470 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

        # evaluate the negative log likelihood *fast*
        %timeit eqx.filter_jit(hessian)(values={"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])}).block_until_ready()
        >> 325 µs ± 2.72 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    ```
    """

    NLL: NLL

    def __init__(self, model: Model, observation: jax.Array) -> None:
        super().__init__(model=model, observation=observation)
        self.NLL = NLL(model=model, observation=observation)

    def __call__(self, values: dict[str, jax.Array] = {}) -> jax.Array:
        if not values:
            values = self.model.parameter_values
        hessian = jax.hessian(self.NLL, argnums=0)(values)
        hessian, _ = jax.tree_util.tree_flatten(hessian)
        hessian = jnp.array(hessian)
        new_shape = len(values)
        return jnp.reshape(hessian, (new_shape, new_shape))


class CovMatrix(Hessian):
    """
    Covariance Matrix.

    Example:
    ```
        from dilax.model import Model
        from dilax.parameter import r, lnN

        class MyModel(Model):
            def evaluate(self) -> EvaluationResult:
                expectations = {}
                # signal
                signal, mu_penalty = self.parameters["mu"](self.processes["signal"], type="r")
                expectations["signal"] = signal

                # background
                background, sigma_penalty = self.parameters["sigma"](self.processes["background"], type="lnN", width=1.1)
                expectations["background"] = background
                return EvaluationResult(expectations=expectations, penalty=mu_penalty + sigma_penalty)

        model = MyModel(
            processes={"signal": jnp.array([10]), "background": jnp.array([50])},
            parameters={"mu": Parameter(value=1.0, bounds=(0, 100)), "sigma": Parameter(value=0, bounds=(-100, 100))},
        )
        observation = jnp.array([60])

        covmatrix = CovMatrix(model=model, observation=observation)

        # evaluate the negative log likelihood
        %timeit covmatrix(values={"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])})
        >> 19 ms ± 504 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

        # evaluate the negative log likelihood *fast*
        %timeit eqx.filter_jit(covmatrix)(values={"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])}).block_until_ready()
        >> 327 µs ± 1.78 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    ```
    """

    def __call__(self, values: dict[str, jax.Array] = {}) -> jax.Array:
        hessian = super().__call__(values=values)
        return jnp.linalg.inv(-hessian)


class SampleToy(BaseModule):
    """
    Sample a toy from the model.

    Example:
    ```
        from dilax.model import Model
        from dilax.parameter import r, lnN

        class MyModel(Model):
            def evaluate(self) -> EvaluationResult:
                expectations = {}
                # signal
                signal, mu_penalty = self.parameters["mu"](self.processes["signal"], type="r")
                expectations["signal"] = signal

                # background
                background, sigma_penalty = self.parameters["sigma"](self.processes["background"], type="lnN", width=1.1)
                expectations["background"] = background
                return EvaluationResult(expectations=expectations, penalty=mu_penalty + sigma_penalty)

        model = MyModel(
            processes={"signal": jnp.array([10]), "background": jnp.array([50])},
            parameters={"mu": Parameter(value=1.0, bounds=(0, 100)), "sigma": Parameter(value=0, bounds=(-100, 100))},
        )
        observation = jnp.array([60])

        sample_toy = SampleToy(model=model, observation=observation)

        values = {"mu": jnp.array([1.1]), "sigma": jnp.array([0.8])}

        # sample a single toy
        toy = sample_toy(values=values, key=jax.random.PRNGKey(1234))

        # sample a single toy *fast*
        toy = eqx.filter_jit(sample_toy)(values=values, key=jax.random.PRNGKey(1234))

        # sample 10 toys
        keys = jax.random.split(jax.random.PRNGKey(1234), num=10)
        toys = eqx.filter_vmap(in_axes=(None, 0))(eqx.filter_jit(sample_toy))(values, keys)

        # new model from toy
        toy = eqx.filter_jit(sample_toy)(values=values, key=jax.random.PRNGKey(1234))
        new_model = model.update(processes=toy)
    ```
    """

    CovMatrix: CovMatrix

    def __init__(self, model: Model, observation: jax.Array) -> None:
        super().__init__(model=model, observation=observation)
        self.CovMatrix = CovMatrix(model=model, observation=observation)

    def __call__(
        self, values: dict[str, jax.Array] = {}, key: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        if key is None:
            key = jax.random.PRNGKey(1234)
        if not values:
            values = self.model.parameter_values
        cov = self.CovMatrix(values=values)
        _values, tree_def = jax.tree_util.tree_flatten(
            self.model.update(values=values).parameter_values
        )
        sampled_values = jax.random.multivariate_normal(
            key=key,
            mean=jnp.concatenate(_values),
            cov=cov,
        )
        new_values = jax.tree_util.tree_unflatten(tree_def, sampled_values)
        model = self.model.update(values=new_values)
        return model.evaluate().expectations
