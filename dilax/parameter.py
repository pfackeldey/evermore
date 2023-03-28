import chex
import jax
import jax.numpy as jnp


@chex.dataclass
class Parameter:
    strength: jax.Array
    bounds: tuple[jnp.array, jnp.array]

    @property
    def boundary_constraint(self) -> jax.Array:
        return jnp.where(
            (self.strength < self.bounds[0]) | (self.strength > self.bounds[1]),
            jnp.inf,
            0,
        )

    @property
    def default_scan_points(self) -> jax.Array:
        ...

    @property
    def logpdf(self) -> jax.Array:
        ...

    def apply(self, sumw: jax.Array) -> jax.Array:
        ...


@chex.dataclass
class FreeFloating(Parameter):
    @property
    def default_scan_points(self) -> jax.Array:
        return jnp.linspace(self.bounds[0], self.bounds[1], 100)

    @property
    def logpdf(self) -> jax.Array:
        return jnp.array(0.0)

    def apply(self, sumw: jax.Array) -> jax.Array:
        return self.strength * sumw


@chex.dataclass
class Normal(Parameter):
    width: jax.Array

    @property
    def default_scan_points(self) -> jax.Array:
        return jnp.linspace(self.strength - 7 * self.width, self.strength + 7 * self.width, 100)

    @property
    def logpdf(self) -> jax.Array:
        return jax.scipy.stats.norm.logpdf(self.strength, loc=1, scale=self.width)

    def apply(self, sumw: jax.Array) -> jax.Array:
        return self.strength * sumw


@chex.dataclass
class LogNormal(Parameter):
    width: jax.Array | tuple[jax.Array, jax.Array]

    @property
    def default_scan_points(self) -> jax.Array:
        if isinstance(self.width, tuple):
            down, up = self.width
        else:
            down = up = self.width
        return jnp.linspace(self.strength - 7 * down, self.strength + 7 * up, 100)

    @property
    def logpdf(self) -> jax.Array:
        if isinstance(self.width, tuple):
            down, up = self.width
            scale = jnp.where(self.strength > 0, up, down)
        else:
            scale = self.width
        return jax.scipy.stats.norm.logpdf(self.strength, loc=0, scale=scale)

    def apply(self, sumw: jax.Array) -> jax.Array:
        return jnp.exp(self.strength) * sumw


# shorthands
from functools import partial

r = partial(FreeFloating, bounds=(jnp.array(-jnp.inf), jnp.array(jnp.inf)))
lnN = partial(LogNormal, bounds=(jnp.array(-jnp.inf), jnp.array(jnp.inf)))


def add_mc_stats(processes, treshold, prefix="mcstat"):
    parameters = {}
    templ_per_process = prefix + "_{}_{}"
    templ_total = prefix + "_{}"
    sumw_total = jnp.sum(jnp.array(list(p[0] for p in processes.values())), axis=0)
    sumw2_total = jnp.sum(jnp.array(list(p[1] for p in processes.values())), axis=0)

    for i in range(sumw_total.shape[0]):
        if sumw_total[i] >= treshold:
            param_name = templ_total.format(i)
            parameters[param_name] = Normal(
                strength=jnp.array(1.0),
                width=jnp.sqrt(sumw2_total[i]) / sumw_total[i],
                bounds=(jnp.array(0.0), jnp.array(jnp.inf)),
            )
        else:
            # per process
            for process, (sumw, sumw2) in processes.items():
                param_name = templ_per_process.format(process, i)
                parameters[param_name] = Normal(
                    strength=jnp.array(1.0),
                    width=jnp.sqrt(sumw2[i]) / sumw[i],
                    bounds=(jnp.array(0.0), jnp.array(jnp.inf)),
                )
    return parameters
