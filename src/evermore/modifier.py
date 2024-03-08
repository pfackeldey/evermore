from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from evermore.custom_types import AddOrMul
from evermore.effect import (
    DEFAULT_EFFECT,
)
from evermore.parameter import Parameter

if TYPE_CHECKING:
    from evermore.effect import Effect

__all__ = [
    "Modifier",
    "compose",
    "where",
]


def __dir__():
    return __all__


class Modifier(eqx.Module):
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

    def scale_factor(self, sumw: Array) -> Array:
        return self.effect.scale_factor(parameter=self.parameter, sumw=sumw)

    @jax.named_scope("evm.modifier")
    def __call__(self, sumw: Array) -> Array:
        op = self.effect.apply_op
        shift = jnp.atleast_1d(self.scale_factor(sumw=sumw))
        shift = jnp.broadcast_to(shift, sumw.shape)
        return op(shift, sumw)  # type: ignore[call-arg]

    def __matmul__(self, other: Composable) -> compose:
        return compose(self, other)


class where(eqx.Module):
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

    def scale_factor(self, sumw: Array) -> Array:
        return jnp.where(
            self.condition,
            self.modifier_true.scale_factor(sumw),
            self.modifier_false.scale_factor(sumw),
        )

    @jax.named_scope("evm.where")
    def __call__(self, sumw: Array) -> Array:
        op_true = self.modifier_true.effect.apply_op
        op_false = self.modifier_false.effect.apply_op
        sf = self.scale_factor(sumw=sumw)
        return jnp.where(
            self.condition,
            op_true(jnp.atleast_1d(sf), sumw),  # type: ignore[call-arg]
            op_false(jnp.atleast_1d(sf), sumw),  # type: ignore[call-arg]
        )

    def __matmul__(self, other: Composable) -> compose:
        return compose(self, other)


class compose(eqx.Module):
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

    modifiers: list[Composable]

    def __init__(self, *modifiers: Composable) -> None:
        self.modifiers = list(modifiers)
        # unroll nested compositions
        _modifiers = []
        for mod in self.modifiers:
            if isinstance(mod, compose):
                _modifiers.extend(mod.modifiers)
            else:
                assert isinstance(mod, Modifier | where)
                _modifiers.append(mod)
        # by now all modifiers are either modifier or staterror
        self.modifiers = _modifiers

    def __len__(self) -> int:
        return len(self.modifiers)

    @jax.named_scope("evm.compose")
    def __call__(self, sumw: Array) -> Array:
        def _prep_shift(modifier: Modifier | where, sumw: Array) -> Array:
            shift = modifier.scale_factor(sumw=sumw)
            shift = jnp.atleast_1d(shift)
            return jnp.broadcast_to(shift, sumw.shape)

        # collect all multiplicative and additive shifts
        shifts: dict[AddOrMul, list] = {operator.mul: [], operator.add: []}
        for m in range(len(self)):
            mod = self.modifiers[m]
            # cast to modifier | staterror, we know it is one of them
            # because we unrolled nested compositions in __init__
            mod = cast(Modifier | where, mod)
            sf = _prep_shift(mod, sumw)
            if isinstance(mod, Modifier):
                if mod.effect.apply_op is operator.mul:
                    shifts[operator.mul].append(sf)
                elif mod.effect.apply_op is operator.add:
                    shifts[operator.add].append(sf)
                else:
                    msg = f"Unsupported apply_op {mod.effect.apply_op} for Modifier {mod}. Only multiplicative and additive effects are supported."
                    raise ValueError(msg)
            elif isinstance(mod, where):
                op_true = mod.modifier_true.effect.apply_op
                op_false = mod.modifier_false.effect.apply_op
                # if both modifiers are multiplicative:
                if op_true is operator.mul and op_false is operator.mul:
                    shifts[operator.mul].append(sf)
                # if both modifiers are additive:
                elif op_true is operator.add and op_false is operator.add:
                    shifts[operator.add].append(sf)
                # if one is multiplicative and the other is additive:
                elif op_true is operator.mul and op_false is operator.add:
                    _mult_sf = jnp.where(mod.condition, sf, 1.0)
                    _add_sf = jnp.where(mod.condition, sf, 0.0)
                    shifts[operator.mul].append(_mult_sf)
                    shifts[operator.add].append(_add_sf)
                elif op_true is operator.add and op_false is operator.mul:
                    _mult_sf = jnp.where(mod.condition, 1.0, sf)
                    _add_sf = jnp.where(mod.condition, 0.0, sf)
                    shifts[operator.mul].append(_mult_sf)
                    shifts[operator.add].append(_add_sf)
                else:
                    msg = f"Unsupported apply_op {op_true} and {op_false} for 'where' Modifier {mod}. Only multiplicative and additive effects are supported."
                    raise ValueError(msg)
        # calculate the product with for operator.mul
        _mult_fact = reduce(operator.mul, shifts[operator.mul], 1.0)
        # calculate the sum for operator.add
        _add_shift = reduce(operator.add, shifts[operator.add], 0.0)
        # apply
        return _mult_fact * (sumw + _add_shift)

    def __matmul__(self, other: Composable) -> compose:
        return compose(self, other)


Composable = Modifier | compose | where
