from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from dilax.effect import (
    DEFAULT_EFFECT,
    gauss,
    poisson,
)
from dilax.pdf import HashablePDF
from dilax.util import as1darray

if TYPE_CHECKING:
    from dilax.effect import Effect

__all__ = [
    "Parameter",
    "modifier",
    "staterror",
    "compose",
]


def __dir__():
    return __all__


class Parameter(eqx.Module):
    value: jax.Array = eqx.field(converter=as1darray)
    bounds: tuple[jax.Array, jax.Array] = eqx.field(
        static=True, converter=lambda x: tuple(map(as1darray, x))
    )
    constraints: set[HashablePDF] = eqx.field(static=True)

    def __init__(
        self,
        value: jax.Array,
        bounds: tuple[jax.Array, jax.Array] = (as1darray(-jnp.inf), as1darray(jnp.inf)),
    ) -> None:
        self.value = value
        self.bounds = bounds
        self.constraints: set[HashablePDF] = set()

    def update(self, value: jax.Array) -> Parameter:
        return eqx.tree_at(lambda t: t.value, self, value)

    @property
    def boundary_penalty(self) -> jax.Array:
        return jnp.where(
            (self.value < self.bounds[0]) | (self.value > self.bounds[1]),
            jnp.inf,
            0,
        )


class ModifierBase(eqx.Module):
    @abc.abstractmethod
    def scale_factor(self, sumw: jax.Array) -> jax.Array:
        ...

    def __call__(self, sumw: jax.Array) -> jax.Array:
        return jnp.atleast_1d(self.scale_factor(sumw=sumw)) * sumw


class modifier(ModifierBase):
    """
    Create a new modifier for a given parameter and penalty.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import dilax as dlx

        mu = dlx.Parameter(value=1.1, bounds=(0, 100))
        norm = dlx.Parameter(value=0.0, bounds=(-jnp.inf, jnp.inf))

        # create a new parameter and a penalty
        modify = dlx.modifier(name="mu", parameter=mu, effect=dlx.effect.unconstrained())

        # apply the modifier
        modify(jnp.array([10, 20, 30]))
        # -> Array([11., 22., 33.], dtype=float32, weak_type=True),

        # lnN effect
        modify = dlx.modifier(name="norm", parameter=norm, effect=dlx.effect.lnN(0.2))
        modify(jnp.array([10, 20, 30]))

        # poisson effect
        hist = jnp.array([10, 20, 30])
        modify = dlx.modifier(name="norm", parameter=norm, effect=dlx.effect.poisson(hist))
        modify(jnp.array([10, 20, 30]))

        # shape effect
        up = jnp.array([12, 23, 35])
        down = jnp.array([8, 19, 26])
        modify = dlx.modifier(name="norm", parameter=norm, effect=dlx.effect.shape(up, down))
        modify(jnp.array([10, 20, 30]))
    """

    name: str
    parameter: Parameter
    effect: Effect

    def __init__(
        self, name: str, parameter: Parameter, effect: Effect = DEFAULT_EFFECT
    ) -> None:
        self.name = name
        self.parameter = parameter
        self.effect = effect
        self.parameter.constraints.add(self.effect.constraint)

    def scale_factor(self, sumw: jax.Array) -> jax.Array:
        return self.effect.scale_factor(parameter=self.parameter, sumw=sumw)


class staterror(ModifierBase):
    """
    Create a staterror (barlow-beeston) modifier which acts on each bin with a different _underlying_ modifier.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import dilax as dlx

        hist = jnp.array([10, 20, 30])

        p1 = dlx.Parameter(value=1.0)
        p2 = dlx.Parameter(value=0.0)
        p3 = dlx.Parameter(value=0.0)

        # all bins with bin content below 10 (threshold) are treated as poisson, else gauss
        modify = dlx.staterror(
            parameters=[p1, p2, p3],
            sumw=hist,
            sumw2=hist,
            threshold=10.0,
        )
        modify(hist)
        # -> Array([13.162277, 20.      , 30.      ], dtype=float32)

        # jit
        import equinox as eqx

        fast_modify = eqx.filter_jit(modify)
    """

    name: str = "staterror"
    parameters: list[Parameter]
    sumw: jax.Array
    sumw2: jax.Array
    sumw2sqrt: jax.Array
    widths: jax.Array
    mask: jax.Array
    threshold: float

    def __init__(
        self,
        parameters: list[Parameter],
        sumw: jax.Array,
        sumw2: jax.Array,
        threshold: float,
    ) -> None:
        self.parameters = parameters
        self.sumw = sumw
        self.sumw2 = sumw2
        self.sumw2sqrt = jnp.sqrt(sumw2)
        self.threshold = threshold

        # calculate width
        self.widths = self.sumw2sqrt / self.sumw

        # store if sumw is below threshold
        self.mask = self.sumw < self.threshold

        for i, param in enumerate(self.parameters):
            effect = poisson(self.sumw[i]) if self.mask[i] else gauss(self.widths[i])
            param.constraints.add(effect.constraint)

    def __check_init__(self):
        if not len(self.parameters) == len(self.sumw2) == len(self.sumw):
            msg = (
                f"Length of parameters ({len(self.parameters)}), "
                f"sumw2 ({len(self.sumw2)}) and sumw ({len(self.sumw)}) "
                "must be the same."
            )
            raise ValueError(msg)

    def scale_factor(self, sumw: jax.Array) -> jax.Array:
        from functools import partial

        assert len(sumw) == len(self.parameters) == len(self.sumw2)

        values = jnp.concatenate([param.value for param in self.parameters])
        idxs = jnp.arange(len(sumw))

        # sumw where mask (poisson) else widths (gauss)
        _widths = jnp.where(self.mask, self.sumw, self.widths)

        def _mod(
            value: jax.Array,
            width: jax.Array,
            idx: jax.Array,
            effect: Effect,
        ) -> jax.Array:
            return effect(width).scale_factor(
                parameter=Parameter(value=value),
                sumw=sumw[idx],
            )[0]

        _poisson_mod = partial(_mod, effect=poisson)
        _gauss_mod = partial(_mod, effect=gauss)

        # where mask use poisson else gauss
        return jnp.where(
            self.mask,
            jax.vmap(_poisson_mod)(values, _widths, idxs),
            jax.vmap(_gauss_mod)(values, _widths, idxs),
        )


class compose(ModifierBase):
    """
    Composition of multiple modifiers, i.e.: `(f ∘ g ∘ h)(hist) = f(hist) * g(hist) * h(hist)`
    It behaves like a single modifier, but it is composed of multiple modifiers; it can be arbitrarly nested.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import dilax as dlx

        mu = dlx.Parameter(value=1.1, bounds=(0, 100))
        sigma = dlx.Parameter(value=0.1, bounds=(-100, 100))

        # create a new parameter and a composition of modifiers
        composition = dlx.compose(
            dlx.modifier(name="mu", parameter=mu),
            dlx.modifier(name="sigma1", parameter=sigma, effect=dlx.effect.lnN(0.1)),
        )

        # apply the composition
        composition(jnp.array([10, 20, 30]))

        # nest compositions
        composition = dlx.compose(
            composition,
            dlx.modifier(name="sigma2", parameter=sigma, effect=dlx.effect.lnN(0.2)),
        )

        # jit
        import equinox as eqx

        eqx.filter_jit(composition)(jnp.array([10, 20, 30]))
    """

    modifiers: tuple[modifier, ...]
    names: list[str] = eqx.field(static=True)

    def __init__(self, *modifiers: modifier) -> None:
        self.modifiers = modifiers

        # set names
        self.names = []
        for m in range(len(self)):
            modifier = self.modifiers[m]
            if isinstance(modifier, compose):
                self.names.extend(modifier.names)
            else:
                self.names.append(modifier.name)

    def __check_init__(self):
        # check for duplicate names
        duplicates = [name for name in self.names if self.names.count(name) > 1]
        if duplicates:
            msg = f"Modifiers need to have unique names, got: {duplicates}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.modifiers)

    def scale_factors(self, sumw: jax.Array) -> dict[str, jax.Array]:
        sfs = {}
        for m in range(len(self)):
            modifier = self.modifiers[m]
            if isinstance(modifier, compose):
                sfs.update(modifier.scale_factors(sumw=sumw))
            else:
                sf = jnp.atleast_1d(modifier.scale_factor(sumw=sumw))
                sfs[modifier.name] = jnp.broadcast_to(sf, sumw.shape)
        return sfs

    def scale_factor(self, sumw: jax.Array) -> jax.Array:
        sfs = jnp.stack(list(self.scale_factors(sumw=sumw).values()))
        # calculate the product in log-space for numerical precision
        return jnp.exp(jnp.sum(jnp.log(sfs), axis=0))
