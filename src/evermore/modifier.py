from __future__ import annotations

import abc
import operator
from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from evermore.custom_types import SF, AddOrMul, ModifierLike
from evermore.effect import DEFAULT_EFFECT
from evermore.parameter import Parameter

if TYPE_CHECKING:
    from evermore.effect import Effect

__all__ = [
    "ModifierBase",
    "Modifier",
    "compose",
    "where",
    "mask",
    "transform",
]


def __dir__():
    return __all__


class AbstractModifier(eqx.Module):
    @abc.abstractmethod
    def scale_factor(self: ModifierLike, sumw: Array) -> SF:
        ...

    @abc.abstractmethod
    def __call__(self: ModifierLike, sumw: Array) -> Array:
        ...

    @abc.abstractmethod
    def __matmul__(self: ModifierLike, other: ModifierLike) -> compose:
        ...


class ApplyFn(eqx.Module):
    @jax.named_scope("evm.modifier.ApplyFn")
    def __call__(self: ModifierLike, sumw: Array) -> Array:
        sf = self.scale_factor(sumw=sumw)
        # apply
        return sf.multiplicative * (sumw + sf.additive)


class MatMulCompose(eqx.Module):
    def __matmul__(self: ModifierLike, other: ModifierLike) -> compose:
        return compose(self, other)


class ModifierBase(ApplyFn, MatMulCompose, AbstractModifier):
    """
    This serves as a base class for all modifiers.
    It automatically implements the __call__ method to apply the scale factors to the sumw array
    and the __matmul__ method to compose two modifiers.

    Custom modifiers should inherit from this class and implement the scale_factor method.

    Example:

        .. code-block:: python

            import equinox as eqx
            import jax.numpy as jnp
            import jax.tree_util as jtu
            from jaxtyping import Array

            import evermore as evm

            class clip(evm.ModifierBase):
                modifier: evm.ModifierBase
                min_sf: float = eqx.field(static=True)
                max_sf: float = eqx.field(static=True)

                def scale_factor(self, sumw: Array) -> evm.custrom_types.SF:
                    sf = self.modifier.scale_factor(sumw)
                    return jtu.tree_map(lambda x: jnp.clip(x, self.min_sf, self.max_sf), sf)


            parameter = evm.Parameter(value=1.1)
            modifier = parameter.unconstrained()

            clipped_modifier = clip(modifier=modifier, min_sf=0.8, max_sf=1.2)

            # this example is trivial, because you can also implement it with `evm.modifier.transform`:
            from functools import partial

            clipped_modifier = evm.modifier.transform(partial(jnp.clip, a_min=0.8, a_max=1.2), modifier)
    """


class Modifier(ModifierBase):
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

    def scale_factor(self, sumw: Array) -> SF:
        return self.effect.scale_factor(parameter=self.parameter, sumw=sumw)


class where(ModifierBase):
    """
    Combine two modifiers based on a condition.

    The condition is a boolean array, and the two modifiers are applied to the data based on the condition.

    Example:

        .. code-block:: python

            import jax.numpy as jnp
            import evermore as evm

            hist = jnp.array([5, 20, 30])
            syst = evm.Parameter(value=0.1)

            norm = syst.lnN(jnp.array([0.9, 1.1]))
            shape = syst.shape(up=jnp.array([7, 22, 31]), down=jnp.array([4, 16, 27]))

            # apply norm if hist < 10, else apply shape
            modifier = evm.modifier.where(hist < 10, norm, shape)

            # apply
            modifier(hist)
            # -> Array([ 5.049494, 20.281374, 30.181376], dtype=float32)

            # for comparison:
            norm(hist)
            # -> Array([ 5.049494, 20.197975, 30.296963], dtype=float32)
            shape(hist)
            # -> Array([ 5.1593127, 20.281374 , 30.181376 ], dtype=float32)
    """

    condition: Array = eqx.field(static=True)
    modifier_true: Modifier
    modifier_false: Modifier

    def scale_factor(self, sumw: Array) -> SF:
        true_sf = self.modifier_true.scale_factor(sumw)
        false_sf = self.modifier_false.scale_factor(sumw)

        def _where(true: Array, false: Array) -> Array:
            return jnp.where(self.condition, true, false)

        return jtu.tree_map(_where, true_sf, false_sf)


class mask(ModifierBase):
    """
    Mask a modifier for specific bins.

    The mask is a boolean array (True, False for each bin).
    The modifier is only applied to the bins where the mask is True.

    Example:

        .. code-block:: python

            import jax.numpy as jnp
            import evermore as evm

            hist = jnp.array([5, 20, 30])
            syst = evm.Parameter(value=0.1)

            norm = syst.lnN(jnp.array([0.9, 1.1]))
            mask = jnp.array([True, False, True])

            modifier = evm.modifier.mask(mask, norm)

            # apply
            modifier(hist)
            # -> Array([ 5.049494, 20.      , 30.296963], dtype=float32)
    """

    where: Array = eqx.field(static=True)
    modifier: Modifier

    def scale_factor(self, sumw: Array) -> SF:
        sf = self.modifier.scale_factor(sumw)

        def _mask(true: Array, false: Array) -> Array:
            return jnp.where(self.where, true, false)

        return SF(
            multiplicative=_mask(sf.multiplicative, jnp.ones_like(sumw)),
            additive=_mask(sf.additive, jnp.zeros_like(sumw)),
        )


class transform(ModifierBase):
    """
    Transform the scale factors of a modifier.

    The `transform_fn` is a function that is applied to both, multiplicative and additive scale factors.

    Example:

        .. code-block:: python

            import jax.numpy as jnp
            import evermore as evm

            hist = jnp.array([5, 20, 30])
            syst = evm.Parameter(value=0.1)

            norm = syst.lnN(jnp.array([0.9, 1.1]))

            transformed_norm = evm.modifier.transform(jnp.sqrt, norm)

            # apply
            transformed_norm(hist)
            # -> Array([ 5.024686, 20.098743, 30.148115], dtype=float32)

            # for comparison:
            norm(hist)
            # -> Array([ 5.049494, 20.197975, 30.296963], dtype=float32)
    """

    transform_fn: Callable = eqx.field(static=True)
    modifier: Modifier

    def scale_factor(self, sumw: Array) -> SF:
        sf = self.modifier.scale_factor(sumw)
        return jtu.tree_map(self.transform_fn, sf)


class compose(ModifierBase):
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
                assert isinstance(mod, ModifierBase)
                _modifiers.append(mod)
        # by now all modifiers are either modifier or staterror
        self.modifiers = _modifiers

    def __len__(self) -> int:
        return len(self.modifiers)

    def scale_factor(self, sumw: Array) -> SF:
        # collect all multiplicative and additive shifts
        sfs: dict[AddOrMul, list] = {operator.add: [], operator.mul: []}
        for m in range(len(self)):
            mod = self.modifiers[m]
            _sf = mod.scale_factor(sumw)
            sfs[operator.mul].append(_sf.multiplicative)
            sfs[operator.add].append(_sf.additive)

        # calculate the product with for operator.mul and operator.add
        multiplicative_sf = reduce(operator.mul, sfs[operator.mul], jnp.ones_like(sumw))
        additive_sf = reduce(operator.add, sfs[operator.add], jnp.zeros_like(sumw))
        return SF(multiplicative=multiplicative_sf, additive=additive_sf)
