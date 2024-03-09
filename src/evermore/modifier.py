from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from evermore.custom_types import AddOrMul, AddOrMulSFs, ModifierLike
from evermore.effect import DEFAULT_EFFECT
from evermore.parameter import Parameter
from evermore.util import initSF

if TYPE_CHECKING:
    from evermore.effect import Effect

__all__ = [
    "Modifier",
    "compose",
    "where",
]


def __dir__():
    return __all__


class ApplyFn(eqx.Module):
    @jax.named_scope("evm.modifier.ApplyFn")
    def __call__(self: ModifierLike, sumw: Array) -> Array:
        sf = self.scale_factor(sumw=sumw)
        # apply
        return sf[operator.mul] * (sumw + sf[operator.add])


class Modifier(ApplyFn):
    """
    Create a new modifier for a given parameter and penalty.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        mu = evm.Parameter(value=1.1)
        norm = evm.Parameter(value=0.0)

        # create a new parameter and a penalty
        modify = evm.modifier(parameter=mu, effect=evm.effect.unconstrained())
        # or shorthand
        modify = mu.unconstrained()

        # apply the modifier
        modify(jnp.array([10, 20, 30]))
        # -> Array([11., 22., 33.], dtype=float32, weak_type=True),

        # lnN effect
        modify = evm.modifier(parameter=norm, effect=evm.effect.lnN(jnp.array([0.8, 1.2])))
        # or shorthand
        modify = norm.lnN(jnp.array([0.8, 1.2]))

        # poisson effect
        hist = jnp.array([10, 20, 30])
        modify = evm.modifier(parameter=norm, effect=evm.effect.poisson(hist))
        # or shorthand
        modify = norm.poisson(hist)

        # shape effect
        up = jnp.array([12, 23, 35])
        down = jnp.array([8, 19, 26])
        modify = evm.modifier(parameter=norm, effect=evm.effect.shape(up, down))
        # or shorthand
        modify = norm.shape(up, down)
    """

    parameter: Parameter
    effect: Effect

    def __init__(self, parameter: Parameter, effect: Effect = DEFAULT_EFFECT) -> None:
        self.parameter = parameter
        self.effect = effect

        # first time: set the constraint pdf
        constraint = self.effect.constraint(parameter=self.parameter)
        self.parameter._set_constraint(constraint, overwrite=False)

    def scale_factor(self, sumw: Array) -> AddOrMulSFs:
        return self.effect.scale_factor(parameter=self.parameter, sumw=sumw)

    def __matmul__(self, other: ModifierLike) -> compose:
        return compose(self, other)


class where(ApplyFn):
    """
    Combine two modifiers based on a condition.

    The condition is a boolean array, and the two modifiers are applied to the data based on the condition.

    Example:

        .. code-block:: python

            import jax.numpy as jnp
            import evermore as evm

            hist = jnp.array([5, 20, 30])
            syst = evm.Parameter(value=0.0)

            norm = syst.lnN(jnp.array([0.9, 1.1]))
            shape = syst.shape(up=jnp.array([7, 22, 31]), down=jnp.array([4, 16, 27]))

            modifier = evm.modifier.where(hist < 10, norm, shape)

            # apply
            modifier(hist)
    """

    condition: Array = eqx.field(static=True)
    modifier_true: Modifier
    modifier_false: Modifier

    def scale_factor(self, sumw: Array) -> AddOrMulSFs:
        sf = initSF(shape=sumw.shape)

        true_sf = self.modifier_true.scale_factor(sumw)
        false_sf = self.modifier_false.scale_factor(sumw)

        for op in operator.mul, operator.add:
            sf.update(jnp.where(self.condition, true_sf[op], false_sf[op]))
        return sf

    def __matmul__(self, other: ModifierLike) -> compose:
        return compose(self, other)


class compose(ApplyFn):
    """
    Composition of multiple modifiers, i.e.: `(f ∘ g ∘ h)(hist) = f(hist) * g(hist) * h(hist)`
    It behaves like a single modifier, but it is composed of multiple modifiers; it can be arbitrarly nested.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        mu = evm.Parameter(value=1.1)
        sigma = evm.Parameter(value=0.1)
        sigma2 = evm.Parameter(value=0.2)

        hist = jnp.array([10, 20, 30])

        # all bins with bin content below 10 (threshold) are treated as poisson, else gauss

        # create a new parameter and a composition of modifiers
        mu_mod = mu.constrained()
        sigma_mod = sigma.lnN(jnp.array([0.9, 1.1]))
        sigma2_mod = sigma2.lnN(jnp.array([0.95, 1.05]))
        composition = evm.compose(
            mu_mod,
            sigma_mod,
            evm.modifier.where(hist < 15, sigma2_mod, sigma_mod),
        )
        # or shorthand
        composition = mu_mod @ sigma_mod @ evm.modifier.where(hist < 15, sigma2_mod, sigma_mod)

        # apply the composition
        composition(hist)

        # nest compositions
        composition = evm.compose(
            composition,
            evm.modifier(parameter=sigma, effect=evm.effect.lnN(jnp.array([0.8, 1.2]))),
        )

        # jit
        import equinox as eqx

        eqx.filter_jit(composition)(hist)
    """

    modifiers: list[ModifierLike]

    def __init__(self, *modifiers: ModifierLike) -> None:
        self.modifiers = list(modifiers)
        # unroll nested compositions
        _modifiers = []
        for mod in self.modifiers:
            if isinstance(mod, compose):
                _modifiers.extend(mod.modifiers)
            else:
                assert isinstance(mod, ModifierLike)
                _modifiers.append(mod)
        # by now all modifiers are either modifier or staterror
        self.modifiers = _modifiers

    def __len__(self) -> int:
        return len(self.modifiers)

    def scale_factor(self, sumw: Array) -> AddOrMulSFs:
        # collect all multiplicative and additive shifts
        sfs: dict[AddOrMul, list] = {operator.add: [], operator.mul: []}
        for m in range(len(self)):
            mod = self.modifiers[m]
            _sf = mod.scale_factor(sumw)
            for op in operator.add, operator.mul:
                sfs[op].append(_sf[op])

        sf = initSF(shape=sumw.shape)
        # calculate the product with for operator.mul and operator.add
        for op, init_val in (
            (operator.mul, jnp.ones_like(sumw)),
            (operator.add, jnp.zeros_like(sumw)),
        ):
            sf[op] = reduce(op, sfs[op], init_val)
        return sf

    def __matmul__(self, other: ModifierLike) -> compose:
        return compose(self, other)
