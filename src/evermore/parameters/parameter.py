from __future__ import annotations

import typing as tp
from collections.abc import Hashable

from flax import nnx
from jaxtyping import Array, ArrayLike, Float

from evermore.util import float_array

if tp.TYPE_CHECKING:
    from evermore.binned.modifier import H, Modifier
    from evermore.parameters.transform import BaseParameterTransformation
    from evermore.pdf import BasePDF


__all__ = [
    "BaseParameter",
    "NormalParameter",
    "Parameter",
    "V",
]


def __dir__():
    return __all__


V = tp.TypeVar("V", bound=Float[Array, "..."])


class BaseParameter(nnx.Variable[V]):
    def __init__(
        self,
        value: V | ArrayLike = 0.0,
        name: str | None = None,
        lower: V | ArrayLike | None = None,
        upper: V | ArrayLike | None = None,
        frozen: bool = False,
        transform: BaseParameterTransformation | None = None,
        tags: frozenset[Hashable] = frozenset(),
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(value=float_array(value), **kwargs)

        # store other metadata
        self.set_metadata(name=name)

        # boundaries
        if lower is not None:
            lower = float_array(lower)
        if upper is not None:
            upper = float_array(upper)
        self.set_metadata(lower=lower, upper=upper)

        # frozen: if True, the parameter is not updated during optimization
        self.set_metadata(frozen=frozen)

        # transform
        self.set_metadata(transform=transform)

        # tags
        self.set_metadata(tags=tags)

    @property
    def prior(self) -> BasePDF | None:
        """Returns the prior distribution associated with this parameter.

        Returns:
            BasePDF | None: Prior distribution, or ``None`` if no prior is set.
        """
        return None

    # modifier shorthands
    def scale(self, slope: ArrayLike = 1.0, offset: ArrayLike = 0.0) -> Modifier:
        """Creates a linear modifier driven by this parameter.

        Args:
            slope: Multiplicative factor applied to the histogram.
            offset: Additive shift applied to the histogram.

        Returns:
            Modifier: Modifier representing the linear effect.
        """
        from evermore.binned.effect import Linear
        from evermore.binned.modifier import Modifier

        return Modifier(
            value=self.get_value(),
            effect=Linear(slope=slope, offset=offset),
        )


class Parameter(BaseParameter[V]):
    """Generic parameter with optional bounds, priors, and metadata.

    Attributes:
        value: Current parameter value (mutable via ``.value``).
        name: Optional human-readable identifier.
        lower: Optional lower bound enforced via transformations.
        upper: Optional upper bound enforced via transformations.
        prior: Optional prior distribution.
        frozen: Whether the parameter participates in optimisation.
        transform: Optional transformation applied during ``unwrap``/``wrap``.
        tags: Additional metadata tags.

    Examples:
        >>> import evermore as evm
        >>> theta = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
        >>> theta.value
        Array(1., dtype=float32)
    """


class NormalParameter(Parameter[V]):
    """Parameter whose default prior is the standard normal distribution.

    Provides convenience methods for log-normal scaling and template morphing.
    """

    def __init__(
        self,
        value: V | ArrayLike = 0.0,
        name: str | None = None,
        lower: V | ArrayLike | None = None,
        upper: V | ArrayLike | None = None,
        frozen: bool = False,
        transform: BaseParameterTransformation | None = None,
        tags: frozenset[Hashable] = frozenset(),
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(
            value=value,
            name=name,
            lower=lower,
            upper=upper,
            frozen=frozen,
            transform=transform,
            tags=tags,
            **kwargs,
        )

    @property
    def prior(self) -> BasePDF:
        """Returns the standard normal prior distribution for this parameter.

        Returns:
            BasePDF: Standard normal distribution.
        """
        from evermore.pdf import Normal

        return Normal(mean=float_array(0.0), width=float_array(1.0))

    def scale_log_asymmetric(self, up: ArrayLike, down: ArrayLike) -> Modifier:
        """Creates an asymmetric log-normal modifier for this parameter.

        Args:
            up: Scaling factor applied to upward deviations.
            down: Scaling factor applied to downward deviations.

        Returns:
            Modifier: Modifier representing the asymmetric exponential effect.
        """
        from evermore.binned.effect import AsymmetricExponential
        from evermore.binned.modifier import Modifier

        return Modifier(
            value=self.get_value(),
            effect=AsymmetricExponential(up=float_array(up), down=float_array(down)),
        )

    def scale_log_symmetric(self, kappa: ArrayLike) -> Modifier:
        """Creates a symmetric log-normal modifier for this parameter.

        Args:
            kappa: scaling factor

        Returns:
            Modifier: Modifier representing the symmetric exponential effect.
        """
        from evermore.binned.effect import SymmetricExponential
        from evermore.binned.modifier import Modifier

        return Modifier(
            value=self.get_value(),
            effect=SymmetricExponential(kappa=float_array(kappa)),
        )

    def morphing(
        self,
        up_template: H,
        down_template: H,
    ) -> Modifier:
        """Creates a vertical template morphing modifier for this parameter.

        Args:
            up_template: Template used for upward variations.
            down_template: Template used for downward variations.

        Returns:
            Modifier: Modifier modelling the morphing effect.
        """
        from evermore.binned.effect import VerticalTemplateMorphing
        from evermore.binned.modifier import Modifier

        return Modifier(
            value=self.get_value(),
            effect=VerticalTemplateMorphing(
                up_template=up_template, down_template=down_template
            ),
        )
