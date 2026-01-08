from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool

from evermore.binned.effect import H, OffsetAndScale
from evermore.parameters.parameter import V
from evermore.util import tree_stack

if TYPE_CHECKING:
    from evermore.binned.effect import BaseEffect

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


class ModifierBase(nnx.Module):
    """Base class for modules that modify histogram templates.

    Subclasses implement :meth:`offset_and_scale` and automatically gain a
    callable interface as well as support for composition via the matrix
    multiplication operator.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> modifier = evm.Parameter(1.0).scale()
        >>> modifier(jnp.array([10.0, 20.0]))
        Array([10., 20.], dtype=float32)
    """

    @abc.abstractmethod
    def offset_and_scale(self: ModifierBase, hist: H) -> OffsetAndScale: ...

    def __call__(self: ModifierBase, hist: H) -> H:
        os = self.offset_and_scale(hist=hist)
        return os.scale * (hist + os.offset)  # ty:ignore[invalid-return-type]

    def __matmul__(self: ModifierBase, other: ModifierBase) -> Compose:
        return Compose(self, other)


class Modifier(ModifierBase):
    """Pairs a parameter with an effect to build a modifier.

    Args:
        parameter: Parameter instance that provides the nuisance strength.
        effect: Effect describing how the histogram is altered.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> modifier = evm.Modifier(
        ...     value=1.1,
        ...     effect=evm.effect.Linear(offset=0.0, slope=1.0),
        ... )
        >>> modifier(jnp.array([10, 20, 30]))
        Array([11., 22., 33.], dtype=float32)
    """

    def __init__(self, value: V, effect: BaseEffect) -> None:
        self.value = value
        self.effect = effect

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        return self.effect(value=self.value, hist=hist)


class Where(ModifierBase):
    """Chooses between two modifiers based on a boolean condition.

    Args:
        condition: Boolean array indicating where to apply ``modifier_true``.
        modifier_true: Modifier evaluated where ``condition`` is ``True``.
        modifier_false: Modifier evaluated where ``condition`` is ``False``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> hist = jnp.array([5, 20, 30])
        >>> syst = evm.NormalParameter(value=0.1)
        >>> norm = syst.scale_log_asymmetric(up=jnp.array([1.1]), down=jnp.array([0.9]))
        >>> shape = syst.morphing(
        ...     up_template=jnp.array([7, 22, 31]),
        ...     down_template=jnp.array([4, 16, 27]),
        ... )
        >>> modifier = evm.modifier.Where(hist < 10, norm, shape)
        >>> modifier(hist)
        Array([ 5.049494, 20.281374, 30.181376], dtype=float32)
    """

    def __init__(
        self,
        condition: Bool[Array, ...],
        modifier_true: ModifierBase,
        modifier_false: ModifierBase,
    ):
        self.condition = condition
        self.modifier_true = modifier_true
        self.modifier_false = modifier_false

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        true_os = self.modifier_true.offset_and_scale(hist)
        false_os = self.modifier_false.offset_and_scale(hist)

        def _where(
            true: Bool[Array, ...],
            false: Bool[Array, ...],
        ) -> Bool[Array, ...]:
            return jnp.where(self.condition, true, false)

        return jax.tree.map(_where, true_os, false_os)


class BooleanMask(ModifierBase):
    """Applies a modifier only to bins selected by a boolean mask.

    Args:
        mask: Boolean array indicating which bins receive the modifier.
        modifier: Modifier that provides the offsets and scales.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> hist = jnp.array([5, 20, 30])
        >>> syst = evm.NormalParameter(value=0.1)
        >>> norm = syst.scale_log_asymmetric(up=1.1, down=0.9)
        >>> mask = jnp.array([True, False, True])
        >>> modifier = evm.modifier.BooleanMask(mask, norm)
        >>> modifier(hist)
        Array([ 5.049494, 20.      , 30.296963], dtype=float32)
    """

    def __init__(
        self,
        mask: Bool[Array, ...],
        modifier: ModifierBase,
    ):
        self.mask = mask
        self.modifier = modifier

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        os = self.modifier.offset_and_scale(hist)

        def _mask(
            true: Bool[Array, ...],
            false: Bool[Array, ...],
        ) -> Bool[Array, ...]:
            return jnp.where(self.mask, true, false)

        return OffsetAndScale(
            offset=_mask(os.offset, jnp.zeros_like(os.offset)),
            scale=_mask(os.scale, jnp.ones_like(os.offset)),
        ).broadcast()


class Transform(ModifierBase):
    """Applies a transformation to both offset and scale of a modifier.

    Args:
        transform_fn: Callable applied to each leaf of the offset and scale.
        modifier: Modifier supplying the original offset and scale values.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> hist = jnp.array([5, 20, 30])
        >>> syst = evm.NormalParameter(value=0.1)
        >>> norm = syst.scale_log_asymmetric(up=1.1, down=0.9)
        >>> transformed_norm = evm.modifier.Transform(jnp.sqrt, norm)
        >>> transformed_norm(hist)
        Array([ 5.024686, 20.098743, 30.148115], dtype=float32)
    """

    def __init__(self, transform_fn: Callable, modifier: ModifierBase) -> None:
        self.transform_fn = transform_fn
        self.modifier = modifier

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        os = self.modifier.offset_and_scale(hist)
        return jax.tree.map(self.transform_fn, os)


class TransformOffset(ModifierBase):
    """Transforms only the offset component of another modifier.

    Args:
        transform_fn: Callable applied to the offset leaves.
        modifier: Modifier providing the original offset values.
    """

    def __init__(self, transform_fn: Callable, modifier: ModifierBase) -> None:
        self.transform_fn = transform_fn
        self.modifier = modifier

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        os = self.modifier.offset_and_scale(hist)
        return OffsetAndScale(
            offset=self.transform_fn(os.offset), scale=os.scale
        ).broadcast()


class TransformScale(ModifierBase):
    """Transforms only the multiplicative scale component of another modifier.

    Args:
        transform_fn: Callable applied to the scale leaves.
        modifier: Modifier providing the original scale values.
    """

    def __init__(self, transform_fn: Callable, modifier: ModifierBase) -> None:
        self.transform_fn = transform_fn
        self.modifier = modifier

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        os = self.modifier.offset_and_scale(hist)
        return OffsetAndScale(
            offset=os.offset, scale=self.transform_fn(os.scale)
        ).broadcast()


class Compose(ModifierBase):
    """Combines multiple modifiers and applies them sequentially.

    Args:
        *modifiers: Modifiers to compose. They are flattened if nested ``Compose`` instances are provided.

    Examples:
        >>> import jax.numpy as jnp
        >>> import evermore as evm
        >>> mu = evm.Parameter(value=1.1)
        >>> syst = evm.NormalParameter(value=0.1)
        >>> hist = jnp.array([10, 20, 30])
        >>> composition = evm.modifier.Compose(
        ...     mu.scale(offset=0, slope=1),
        ...     syst.scale_log_asymmetric(up=1.1, down=0.9),
        ... )
        >>> composition(hist)
        Array([11.155, 22.237, 33.318], dtype=float32)
    """

    def __init__(self, *modifiers: ModifierBase) -> None:
        if not modifiers:
            msg = "At least one modifier must be provided to Compose."
            raise ValueError(msg)
        # unpack the modifiers, if they are already a Compose instance
        # this allows for nested compositions, e.g.:
        # `Compose(Modifier1, Compose(Modifier2, Modifier3))`
        # will flatten to `Compose(Modifier1, Modifier2, Modifier3)`
        self.modifiers = nnx.List(unroll(modifiers))

    def __len__(self) -> int:
        return len(self.modifiers)

    def offset_and_scale(self, hist: H) -> OffsetAndScale:
        from collections import defaultdict

        # initial scale and offset
        scale = jnp.ones_like(hist)
        offset = jnp.zeros_like(hist)

        groups = defaultdict(list)
        # first group modifiers into same NNX graph structures
        for mod in self.modifiers:
            graphdef, state = nnx.split(mod)
            groups[graphdef].append(state)
        # then do the `vmap` trick to calculate the scale factors in parallel per group.
        for graphdef, states in groups.items():
            # skip empty groups
            if not states:
                continue
            # Essentially we are turning an array of modifiers (AOS) into a single modifier of stacked leaves (SOA).
            # Then we can use XLA's vectorization/loop constructs (e.g.: `jax.vmap` or `jax.lax.scan`) to calculate
            # the scale factors without having to compile the fully unrolled loop.
            dynamic_stack = tree_stack(states, broadcast_leaves=True)

            def calc_sf(_hist, dynamic_stack, graphdef):
                stack = nnx.merge(graphdef, dynamic_stack)
                return stack.offset_and_scale(_hist)

            # Vectorize over the first axis of the stack.
            # Using `jax.vmap` is the most efficient way to do this,
            # however it needs `hist` and `dynamic_stack` to fit into memory.
            # If this is not the case, we should consider using `jax.lax.scan` instead.
            # See: https://github.com/jax-ml/jax/discussions/19114#discussioncomment-7996283
            vec_calc_sf = nnx.vmap(
                jax.tree_util.Partial(calc_sf, graphdef=graphdef),
                in_axes=(None, 0),  # vectorize over the batch axis of the dynamic_stack
                out_axes=0,  # return a tree of scale factors
            )
            os = vec_calc_sf(hist, dynamic_stack)
            scale *= jnp.prod(os.scale, axis=0)
            offset += jnp.sum(os.offset, axis=0)

        return OffsetAndScale(offset=offset, scale=scale).broadcast()


def unroll(modifiers: Iterable[ModifierBase]) -> Iterator[ModifierBase]:
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
