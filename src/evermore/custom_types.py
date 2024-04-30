from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

if TYPE_CHECKING:
    from evermore.modifier import Compose


__all__ = [
    "OffsetAndScale",
    "ModifierLike",
    "PDFLike",
]


class OffsetAndScale(eqx.Module):
    offset: Array = eqx.field(converter=jnp.atleast_1d, default=0.0)
    scale: Array = eqx.field(converter=jnp.atleast_1d, default=1.0)

    def broadcast(self) -> OffsetAndScale:
        shape = jnp.broadcast_shapes(self.offset.shape, self.scale.shape)
        return type(self)(
            offset=jnp.broadcast_to(self.offset, shape),
            scale=jnp.broadcast_to(self.scale, shape),
        )


@runtime_checkable
class ModifierLike(Protocol):
    def offset_and_scale(self, hist: Array) -> OffsetAndScale: ...
    def __call__(self, hist: Array) -> Array: ...
    def __matmul__(self, other: ModifierLike) -> Compose: ...


@runtime_checkable
class PDFLike(Protocol):
    """Mirrors the (relevant) interface of `tfp.distributions.Distribution` & `distrax.Distribution`."""

    def log_prob(self, x: Array) -> Array: ...
    def sample(self, key: PRNGKeyArray) -> Array: ...
