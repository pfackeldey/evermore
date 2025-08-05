from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

import equinox as eqx
from jaxtyping import Array, ArrayLike, Float

from evermore.util import _missing, maybe_float_array
from evermore.visualization import SupportsTreescope

if TYPE_CHECKING:
    from evermore.binned.modifier import H, Modifier
    from evermore.parameters.transform import AbstractParameterTransformation
    from evermore.pdf import AbstractPDF, Normal


__all__ = [
    "AbstractParameter",
    "NormalParameter",
    "Parameter",
    "correlate",
    "replace_value",
]


def __dir__():
    return __all__


def _numeric_methods(name):
    bname = f"__{name}__"

    def _binary(self, other):
        other_val = other.value if isinstance(other, AbstractParameter) else other
        return getattr(self.value, bname)(other_val)

    rname = f"__r{name}__"

    def _reflected(self, other):
        other_val = other.value if isinstance(other, AbstractParameter) else other
        return getattr(self.value, rname)(other_val)

    iname = f"__i{name}__"

    def _inplace(self, other):
        other_val = other.value if isinstance(other, AbstractParameter) else other
        return getattr(self.value, iname)(other_val)

    _binary.__name__ = bname
    _reflected.__name__ = rname
    _inplace.__name__ = iname
    return _binary, _reflected, _inplace


def _unary_method(name):
    uname = f"__{name}__"

    def _unary(self):
        return getattr(self.value, uname)()

    _unary.__name__ = uname
    return _unary


V = TypeVar("V", bound=Float[Array, "..."])


class ValueAttr(eqx.Module, Generic[V], SupportsTreescope):
    value: V


def to_value(x: ArrayLike) -> ValueAttr:
    if isinstance(x, ValueAttr):
        return x
    return ValueAttr(value=maybe_float_array(x, passthrough=False))


class AbstractParameter(
    eqx.Module,
    Generic[V],
    SupportsTreescope,
):
    raw_value: eqx.AbstractVar[V]
    name: eqx.AbstractVar[str | None]
    lower: eqx.AbstractVar[V | None]
    upper: eqx.AbstractVar[V | None]
    prior: eqx.AbstractVar[AbstractPDF | None]
    frozen: eqx.AbstractVar[bool]
    transform: eqx.AbstractVar[AbstractParameterTransformation | None]
    tags: eqx.AbstractVar[frozenset[Hashable]]

    @property
    def value(self) -> V:
        """
        Returns the value of the parameter.

        This property is used to access the actual value of the parameter, which can be a JAX array or any other type.
        It is defined as a property to allow for lazy evaluation and potential transformations.
        """
        return self.raw_value.value

    def __jax_array__(self):
        return self.value

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self) -> Iterator:
        return iter(self.value)

    def __contains__(self, item) -> bool:
        return item in self.value

    __add__, __radd__, __iadd__ = _numeric_methods("add")
    __sub__, __rsub__, __isub__ = _numeric_methods("sub")
    __mul__, __rmul__, __imul__ = _numeric_methods("mul")
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods("matmul")
    __truediv__, __rtruediv__, __itruediv__ = _numeric_methods("truediv")
    __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods("floordiv")
    __mod__, __rmod__, __imod__ = _numeric_methods("mod")
    __divmod__, __rdivmod__, __idivmod__ = _numeric_methods("divmod")
    __pow__, __rpow__, __ipow__ = _numeric_methods("pow")
    __lshift__, __rlshift__, __ilshift__ = _numeric_methods("lshift")
    __rshift__, __rrshift__, __irshift__ = _numeric_methods("rshift")
    __and__, __rand__, __iand__ = _numeric_methods("and")
    __xor__, __rxor__, __ixor__ = _numeric_methods("xor")
    __or__, __ror__, __ior__ = _numeric_methods("or")

    __neg__ = _unary_method("neg")
    __pos__ = _unary_method("pos")
    __abs__ = _unary_method("abs")
    __invert__ = _unary_method("invert")
    __complex__ = _unary_method("complex")
    __int__ = _unary_method("int")
    __float__ = _unary_method("float")
    __index__ = _unary_method("index")
    __trunc__ = _unary_method("trunc")
    __floor__ = _unary_method("floor")
    __ceil__ = _unary_method("ceil")

    def __round__(self, ndigits: int) -> V:
        return self.value.__round__(ndigits)

    def scale(self, slope: ArrayLike = 1.0, offset: ArrayLike = 0.0) -> Modifier:
        """
        Applies a linear scaling effect to the parameter.

        Args:
            slope (ArrayLike, optional): The slope of the linear scaling. Defaults to 1.0.
            offset (ArrayLike, optional): The offset of the linear scaling. Defaults to 0.0.

        Returns:
            Modifier: A Modifier instance with the linear scaling effect applied.
        """
        from evermore.binned.effect import Linear
        from evermore.binned.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=Linear(slope=slope, offset=offset),
        )


class Parameter(AbstractParameter[V]):
    """
    A general Parameter class for defining the parameters of a statistical model.

    Attributes:
        value (V): The actual value of the parameter.
        name (str | None): An optional name for the parameter.
        lower (V | None): The lower boundary of the parameter.
        upper (V | None): The upper boundary of the parameter.
        prior (AbstractPDF | None): The prior distribution of the parameter.
        frozen (bool): Indicates if the parameter is frozen during optimization.
        transform (AbstractParameterTransformation | None): An optional transformation applied to the parameter.
        tags (frozenset[Hashable]): A set of tags associated with the parameter for additional metadata.

    Usage:

    .. code-block:: python

        import evermore as evm

        simple_param = evm.Parameter(value=1.0)
        bounded_param = evm.Parameter(value=1.0, lower=0.0, upper=2.0)
        constrained_parameter = evm.Parameter(
            value=1.0, prior=evm.pdf.Normal(mean=1.0, width=0.1)
        )
        frozen_parameter = evm.Parameter(value=1.0, frozen=True)
    """

    raw_value: V
    name: str | None
    lower: V | None
    upper: V | None
    prior: AbstractPDF | None
    frozen: bool
    transform: AbstractParameterTransformation | None
    tags: frozenset[Hashable] = eqx.field(static=True)

    def __init__(
        self,
        value: V | ArrayLike = 0.0,
        name: str | None = None,
        lower: V | ArrayLike | None = None,
        upper: V | ArrayLike | None = None,
        prior: AbstractPDF | None = None,
        frozen: bool = False,
        transform: AbstractParameterTransformation | None = None,
        tags: frozenset[Hashable] = frozenset(),
    ) -> None:
        self.raw_value = to_value(value)
        self.name = name

        # boundaries
        self.lower = maybe_float_array(lower)
        self.upper = maybe_float_array(upper)

        # prior
        self.prior = prior

        # frozen: if True, the parameter is not updated during optimization
        self.frozen = frozen
        self.transform = transform

        self.tags = tags

    def __check_init__(self):
        from evermore.pdf import AbstractPDF

        # runtime check to be sure
        if self.prior is not None and not isinstance(self.prior, AbstractPDF):
            msg = f"Prior must be a AbstractPDF object for a constrained AbstractParameter (or 'None' for an unconstrained one), got {self.prior=} ({type(self.prior)=})"  # type: ignore[unreachable]
            raise ValueError(msg)


class NormalParameter(AbstractParameter[V]):
    """
    A specialized Parameter class with a Normal prior distribution.

    This class extends the general Parameter class by setting a default Normal prior distribution.
    It also provides additional methods for scaling and morphing the parameter.

    Attributes:
        value (V): The actual value of the parameter.
        name (str | None): An optional name for the parameter.
        lower (V | None): The lower boundary of the parameter.
        upper (V | None): The upper boundary of the parameter.
        prior (Normal): The prior distribution of the parameter, set to a Normal distribution by default.
        frozen (bool): Indicates if the parameter is frozen during optimization.
        transform (AbstractParameterTransformation | None): An optional transformation applied to the parameter.
        tags (frozenset[Hashable]): A set of tags associated with the parameter for additional metadata.

    """

    raw_value: V
    name: str | None
    lower: V | None
    upper: V | None
    prior: Normal
    frozen: bool
    transform: AbstractParameterTransformation | None
    tags: frozenset[Hashable] = eqx.field(static=True)

    def __init__(
        self,
        value: V | ArrayLike = 0.0,
        name: str | None = None,
        lower: V | ArrayLike | None = None,
        upper: V | ArrayLike | None = None,
        frozen: bool = False,
        transform: AbstractParameterTransformation | None = None,
        tags: frozenset[Hashable] = frozenset(),
    ) -> None:
        from evermore.pdf import Normal

        self.raw_value = to_value(value)
        self.name = name

        # boundaries
        self.lower = maybe_float_array(lower)
        self.upper = maybe_float_array(upper)

        # prior
        self.prior = Normal(mean=0.0, width=1.0)

        # frozen: if True, the parameter is not updated during optimization
        self.frozen = frozen
        self.transform = transform

        self.tags = tags

    def scale_log(self, up: ArrayLike, down: ArrayLike) -> Modifier:
        """
        Applies an asymmetric exponential scaling to the parameter.

        Args:
            up (ArrayLike): The scaling factor for the upward direction.
            down (ArrayLike): The scaling factor for the downward direction.

        Returns:
            Modifier: A Modifier instance with the asymmetric exponential effect applied.
        """
        from evermore.binned.effect import AsymmetricExponential
        from evermore.binned.modifier import Modifier

        return Modifier(parameter=self, effect=AsymmetricExponential(up=up, down=down))

    def morphing(
        self,
        up_template: H,
        down_template: H,
    ) -> Modifier:
        """
        Applies vertical template morphing to the parameter.

        Args:
            up_template (H): The template for the upward shift.
            down_template (H): The template for the downward shift.

        Returns:
            Modifier: A Modifier instance with the vertical template morphing effect applied.
        """
        from evermore.binned.effect import VerticalTemplateMorphing
        from evermore.binned.modifier import Modifier

        return Modifier(
            parameter=self,
            effect=VerticalTemplateMorphing(
                up_template=up_template, down_template=down_template
            ),
        )


def replace_value(
    param: AbstractParameter,
    value: V,
) -> AbstractParameter:
    """
    Replaces the `value` attribute of a given `AbstractParameter` instance with a new value.

    Args:
        param (AbstractParameter): The parameter object whose value is to be replaced.
        value (V): The new value to assign to the parameter.

    Returns:
        AbstractParameter: A new `AbstractParameter` instance with the updated value.

    Notes:
        This function uses `eqx.tree_at` to perform a functional update, returning a new object
        rather than modifying the original `param` in place.
    """
    return eqx.tree_at(
        lambda p: p.raw_value,
        param,
        to_value(value),
        is_leaf=lambda leaf: leaf is _missing,
    )


def correlate(*parameters: AbstractParameter) -> tuple[AbstractParameter, ...]:
    """
    Correlate parameters by sharing the value of the *first* given parameter.

    It is preferred to just use the same parameter if possible, this function should be used if that is not doable.

    Args:
        *parameters (AbstractParameter): A variable number of AbstractParameter instances to be correlated.

    Returns:
        tuple[AbstractParameter, ...]: A tuple of correlated AbstractParameter instances.

    Example:

    .. code-block:: python

        import jax
        from jaxtyping import PyTree
        import evermore as evm

        p1 = evm.Parameter(value=1.0)
        p2 = evm.Parameter(value=0.0)
        p3 = evm.Parameter(value=0.5)


        def model(*parameters: PyTree[evm.Parameter]):
            # correlate them inside the model
            p1, p2, p3 = evm.parameter.correlate(*parameters)

            # now p1, p2, p3 are correlated, i.e., they share the same value
            assert p1.value == p2.value == p3.value


        # use the model
        model(p1, p2, p3)

        # More general case of correlating any PyTree of parameters
        from typing import NamedTuple


        class Params(NamedTuple):
            mu: evm.Parameter
            syst: evm.NormalParameter


        params = Params(mu=evm.Parameter(1.0), syst=evm.NormalParameter(0.0))


        def model(params: Params):
            flat_params, tree_def = jax.tree.flatten(params, evm.filter.is_parameter)

            # correlate the parameters
            correlated_flat_params = evm.parameter.correlate(*flat_params)
            correlated_params = jax.tree.unflatten(tree_def, correlated_flat_params)

            # now correlated_params.mu and correlated_params.syst are correlated, i.e., they share the same value
            assert correlated_params.mu.value == correlated_params.syst.value


        # use the model
        model(params)
    """

    first, *rest = parameters

    def _correlate(
        parameter: AbstractParameter[V],
    ) -> tuple[AbstractParameter[V], AbstractParameter[V]]:
        ps = (first, parameter)

        def where(ps: tuple[AbstractParameter[V], AbstractParameter[V]]) -> V:
            return ps[1].value

        def get(ps: tuple[AbstractParameter[V], AbstractParameter[V]]) -> V:
            return ps[0].value

        shared = eqx.nn.Shared(ps, where, get)
        return shared()

    correlated = [first]
    for p in rest:
        # is this error really needed? shouldn't it be safe to broadcast here?
        if p.value.shape != first.value.shape:
            msg = f"Can't correlate parameters {first} and {p}! Must have the same shape, got {first.value.shape} and {p.value.shape}."
            raise ValueError(msg)
        _, p_corr = _correlate(p)
        correlated.append(p_corr)
    return tuple(correlated)
