from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from evermore.binned.effect import H, OffsetAndScale
from evermore.parameters.tree import PT
from evermore.util import tree_stack
from evermore.visualization import SupportsTreescope

if TYPE_CHECKING:
    from evermore.binned.effect import AbstractEffect

__all__ = [
    "BooleanMask",
    "Compose",
    "Modifier",
    "ModifierBase",
    "Transform",
    "TransformOffset",
    "TransformScale",
    "Where",
]


def __dir__():
    return __all__


class AbstractModifier(eqx.Module, Generic[PT], SupportsTreescope):
    @abc.abstractmethod
    def offset_and_scale(self: ModifierBase, hist: H) -> OffsetAndScale[H]: ...

    @abc.abstractmethod
    def __call__(self: ModifierBase, hist: H) -> H: ...

    @abc.abstractmethod
    def __matmul__(self: ModifierBase, other: ModifierBase) -> Compose: ...


class ApplyFn(AbstractModifier[PT]):
    @jax.named_scope("evm.modifier.ApplyFn")
    def __call__(self: ModifierBase, hist: H) -> H:
        os = self.offset_and_scale(hist=hist)
        return os.scale * (hist + os.offset)


class MatMulCompose(AbstractModifier[PT]):
    def __matmul__(self: ModifierBase, other: ModifierBase) -> Compose:
        return Compose(self, other)


class ModifierBase(ApplyFn[PT], MatMulCompose[PT]):
    """
    This serves as a base class for all modifiers.
    It automatically implements the __call__ method to apply the scale factors to the hist array
    and the __matmul__ method to compose two modifiers.

    Custom modifiers should inherit from this class and implement the scale_factor method.

    Example:

    .. code-block:: python

        import equinox as eqx
        import jax.numpy as jnp
        from jaxtyping import Array, Float

        import evermore as evm


        class Clip(evm.modifier.ModifierBase[evm.PT]):
            modifier: evm.modifier.ModifierBase[evm.PT]
            min_sf: float = eqx.field(static=True)
            max_sf: float = eqx.field(static=True)

            def offset_and_scale(
                self, hist: Float[Array, "..."]
            ) -> evm.effect.OffsetAndScale[H]:
                os = self.modifier.offset_and_scale(hist)
                return jax.tree.map(lambda x: jnp.clip(x, self.min_sf, self.max_sf), os)


        parameter = evm.Parameter(value=1.1)
        modifier = parameter.scale()

        clipped_modifier = Clip(modifier=modifier, min_sf=0.8, max_sf=1.2)

        # this example is trivial, because you can also implement it with *evm.modifier.Transform*:
        from functools import partial

        clipped_modifier = evm.modifier.Transform(
            partial(jnp.clip, a_min=0.8, a_max=1.2), modifier
        )
    """


class Modifier(ModifierBase[PT]):
    """
    Create a new modifier for a given parameter and penalty.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        mu = evm.Parameter(value=1.1)
        norm = evm.NormalParameter(value=0.0)

        # create a new parameter and a penalty
        modify = evm.Modifier(parameter=mu, effect=evm.effect.Linear(offset=0, slope=1))
        # or shorthand
        modify = mu.scale(offset=0, slope=1)

        # apply the Modifier
        modify(jnp.array([10, 20, 30]))
        # -> Array([11., 22., 33.], dtype=float32, weak_type=True),

        # log_normal effect
        modify = evm.Modifier(
            parameter=norm, effect=evm.effect.AsymmetricExponential(up=1.2, down=0.8)
        )
        # or shorthand
        modify = norm.scale_log(up=1.2, down=0.8)

        # poisson effect
        hist = jnp.array([10, 20, 30])
        parameter = evm.Parameter(value=1.0, prior=evm.pdf.PoissonDiscrete(lamb=hist))
        modify = evm.Modifier(parameter=parameter, effect=evm.effect.Linear(offset=1, slope=1))
        # or shorthand
        modify = norm.scale(offset=1, slope=1)

        # shape effect
        up_template = jnp.array([12, 23, 35])
        down_template = jnp.array([8, 19, 26])
        modify = evm.Modifier(
            parameter=norm,
            effect=evm.effect.VerticalTemplateMorphing(
                up_template=up_template, down_template=down_template
            ),
        )
        # or shorthand
        modify = norm.morphing(up_template=up_template, down_template=down_template)
    """

    parameter: PT
    effect: AbstractEffect

    def __init__(self, parameter: PT, effect: AbstractEffect[H]) -> None:
        self.parameter = parameter
        self.effect = effect

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        return self.effect(parameter=self.parameter, hist=hist)


class Where(ModifierBase[PT]):
    """
    Combine two modifiers based on a condition.

    The condition is a boolean array, and the two modifiers are applied to the data based on the condition.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        hist = jnp.array([5, 20, 30])
        syst = evm.NormalParameter(value=0.1)

        norm = syst.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
        shape = syst.morphing(
            up_template=jnp.array([7, 22, 31]), down_template=jnp.array([4, 16, 27])
        )

        # apply norm if hist < 10, else apply shape
        modifier = evm.modifier.Where(hist < 10, norm, shape)

        # apply
        modifier(hist)
        # -> Array([ 5.049494, 20.281374, 30.181376], dtype=float32)

        # for comparison:
        norm(hist)
        # -> Array([ 5.049494, 20.197975, 30.296963], dtype=float32)
        shape(hist)
        # -> Array([ 5.1593127, 20.281374 , 30.181376 ], dtype=float32)
    """

    condition: Bool[Array, ...]
    modifier_true: ModifierBase[PT]
    modifier_false: ModifierBase[PT]

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        true_os = self.modifier_true.offset_and_scale(hist)
        false_os = self.modifier_false.offset_and_scale(hist)

        def _where(
            true: Bool[Array, ...],
            false: Bool[Array, ...],
        ) -> Bool[Array, ...]:
            return jnp.where(self.condition, true, false)

        return jax.tree.map(_where, true_os, false_os)


class BooleanMask(ModifierBase[PT]):
    """
    Mask a modifier for specific bins.

    The mask is a boolean array (True, False for each bin).
    The modifier is only applied to the bins where the mask is True.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        hist = jnp.array([5, 20, 30])
        syst = evm.NormalParameter(value=0.1)

        norm = syst.scale_log(up=1.1, down=0.9)
        mask = jnp.array([True, False, True])

        modifier = evm.modifier.BooleanMask(mask, norm)

        # apply
        modifier(hist)
        # -> Array([ 5.049494, 20.      , 30.296963], dtype=float32)
    """

    mask: Bool[Array, ...]
    modifier: ModifierBase[PT]

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        os = self.modifier.offset_and_scale(hist)

        def _mask(
            true: Bool[Array, ...],
            false: Bool[Array, ...],
        ) -> Bool[Array, ...]:
            return jnp.where(self.mask, true, false)

        return OffsetAndScale[H](
            offset=_mask(os.offset, 0.0),
            scale=_mask(os.scale, 1.0),
        ).broadcast()


class Transform(ModifierBase[PT]):
    """
    Transform the scale factors of a modifier.

    The `transform_fn` is a function that is applied to both, multiplicative and additive scale factors.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        hist = jnp.array([5, 20, 30])
        syst = evm.NormalParameter(value=0.1)

        norm = syst.scale_log(up=1.1, down=0.9)

        transformed_norm = evm.modifier.Transform(jnp.sqrt, norm)

        # apply
        transformed_norm(hist)
        # -> Array([ 5.024686, 20.098743, 30.148115], dtype=float32)

        # for comparison:
        norm(hist)
        # -> Array([ 5.049494, 20.197975, 30.296963], dtype=float32)
    """

    transform_fn: Callable = eqx.field(static=True)
    modifier: ModifierBase[PT]

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        os = self.modifier.offset_and_scale(hist)
        return jax.tree.map(self.transform_fn, os)


class TransformOffset(ModifierBase[PT]):
    transform_fn: Callable = eqx.field(static=True)
    modifier: ModifierBase[PT]

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        os = self.modifier.offset_and_scale(hist)
        return OffsetAndScale(
            offset=self.transform_fn(os.offset), scale=os.scale
        ).broadcast()


class TransformScale(ModifierBase[PT]):
    transform_fn: Callable = eqx.field(static=True)
    modifier: ModifierBase[PT]

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        os = self.modifier.offset_and_scale(hist)
        return OffsetAndScale(
            offset=os.offset, scale=self.transform_fn(os.scale)
        ).broadcast()


class Compose(ModifierBase[PT]):
    """
    Composition of multiple modifiers, in order to correctly apply them *together*.
    It behaves like a single modifier, but it is composed of multiple modifiers; it can be arbitrarily nested.

    Example:

    .. code-block:: python

        import jax.numpy as jnp
        import evermore as evm

        mu = evm.Parameter(value=1.1)
        sigma = evm.NormalParameter(value=0.1)
        sigma2 = evm.NormalParameter(value=0.2)

        hist = jnp.array([10, 20, 30])

        # all bins with bin content below 10 (threshold) are treated as poisson, else normal

        # create a new parameter and a composition of modifiers
        mu_mod = mu.scale(offset=0, slope=1)
        sigma_mod = sigma.scale_log(up=1.1, down=0.9)
        sigma2_mod = sigma2.scale_log(up=1.05, down=0.95)
        composition = evm.modifier.Compose(
            mu_mod,
            sigma_mod,
            evm.modifier.Where(hist < 15, sigma2_mod, sigma_mod),
        )
        # or shorthand
        composition = mu_mod @ sigma_mod @ evm.modifier.Where(hist < 15, sigma2_mod, sigma_mod)

        # apply the composition
        composition(hist)

        # nest compositions
        composition = evm.modifier.Compose(
            composition,
            evm.Modifier(
                parameter=sigma, effect=evm.effect.AsymmetricExponential(up=1.2, down=0.8)
            ),
        )

        # jit
        import equinox as eqx

        eqx.filter_jit(composition)(hist)
    """

    modifiers: list[ModifierBase[PT]]

    def __init__(self, *modifiers: ModifierBase[PT]) -> None:
        if not modifiers:
            msg = "At least one modifier must be provided to Compose."
            raise ValueError(msg)
        # unpack the modifiers, if they are already a Compose instance
        # this allows for nested compositions, e.g.:
        # `Compose(Modifier1, Compose(Modifier2, Modifier3))`
        # will flatten to `Compose(Modifier1, Modifier2, Modifier3)`
        self.modifiers = list(unroll(modifiers))

    def __len__(self) -> int:
        return len(self.modifiers)

    def offset_and_scale(self, hist: H) -> OffsetAndScale[H]:
        from collections import defaultdict

        # initial scale and offset
        scale = jnp.ones_like(hist)
        offset = jnp.zeros_like(hist)

        groups = defaultdict(list)
        # first group modifiers into same tree structures
        for mod in self.modifiers:
            groups[hash(jax.tree.structure(mod))].append(mod)
        # then do the `jax.vmap` trick to calculate the scale factors in parallel per group.
        for _, group_mods in groups.items():
            # skip empty groups
            if not group_mods:
                continue
            # Essentially we are turning an array of modifiers (AOS) into a single modifier of stacked leaves (SOA).
            # Then we can use XLA's vectorization/loop constructs (e.g.: `jax.vmap` or `jax.lax.scan`) to calculate
            # the scale factors without having to compile the fully unrolled loop.
            stack = tree_stack(group_mods, broadcast_leaves=True)
            dynamic_stack, static_stack = eqx.partition(stack, eqx.is_array)

            def calc_sf(_hist, _dynamic_stack, _static_stack):
                stack = eqx.combine(_dynamic_stack, _static_stack)
                return stack.offset_and_scale(_hist)

            # Vectorize over the first axis of the stack.
            # Using `jax.vmap` is the most efficient way to do this,
            # however it needs `hist` and `dynamic_stack` to fit into memory.
            # If this is not the case, we should consider using `jax.lax.scan` instead.
            # See: https://github.com/jax-ml/jax/discussions/19114#discussioncomment-7996283
            vec_calc_sf = jax.vmap(
                jax.tree_util.Partial(calc_sf, _static_stack=static_stack),
                in_axes=(None, 0),  # vectorize over the batch axis of the dynamic_stack
                out_axes=0,  # return a tree of scale factors
            )
            os = vec_calc_sf(hist, dynamic_stack)
            scale *= jnp.prod(os.scale, axis=0)
            offset += jnp.sum(os.offset, axis=0)

        return OffsetAndScale(offset=offset, scale=scale).broadcast()


def unroll(modifiers: Iterable[ModifierBase[PT]]) -> Iterator[ModifierBase[PT]]:
    # Helper to recursively flatten nested Compose instances into a single list
    for mod in modifiers:
        if isinstance(mod, Compose):
            # recursively yield from the modifiers of the Compose instance
            yield from unroll(mod.modifiers)
        elif isinstance(mod, ModifierBase):
            # yield the modifier if it is a ModifierBase instance
            yield mod
        else:
            # raise an error if the modifier is not a ModifierBase instance
            msg = f"Modifier {mod} is not a ModifierBase instance."  # type: ignore[unreachable]
            raise TypeError(msg)
