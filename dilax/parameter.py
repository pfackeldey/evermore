from __future__ import annotations

import abc

import equinox as eqx

import jax
import jax.numpy as jnp


class Parameter(eqx.Module):
    value: jax.Array = eqx.field(converter=jax.numpy.asarray)
    bounds: tuple[jnp.array, jnp.array]

    def __init__(
        self,
        value: jax.Array,
        bounds: tuple[jnp.array, jnp.array],
    ) -> None:
        self.value = value
        self.bounds = bounds

    def update(self, value: jax.Array) -> Parameter:
        return self.__class__(value=value, bounds=self.bounds)

    @property
    def boundary_penalty(self) -> jax.Array:
        return jnp.where(
            (self.value < self.bounds[0]) | (self.value > self.bounds[1]),
            jnp.inf,
            0,
        )

    @eqx.filter_jit
    def __call__(self, sumw: jax.Array, type: str, **kwargs) -> tuple[jax.Array, jax.Array]:
        penalty = Penalty.getcls(type)(parameter=self, **kwargs)
        return penalty(sumw=sumw), penalty.logpdf


class Penalty(eqx.Module):
    parameter: Parameter

    def __init__(self, parameter: Parameter) -> None:
        self.parameter = parameter

    @property
    @abc.abstractmethod
    def type(self) -> str:
        ...

    @classmethod
    def gettypes(cls) -> list[str]:
        return [sub_cls.type for sub_cls in cls.__subclassess__()]

    @classmethod
    def getcls(cls, type: str) -> Penalty:
        for sub_cls in cls.__subclasses__():
            if sub_cls.type == type:
                return sub_cls
        raise ValueError(f"Unknown penalty type: {type}, available: {cls.gettypes()}")

    @property
    @abc.abstractmethod
    def default_scan_points(self) -> jax.Array:
        ...

    @property
    @abc.abstractmethod
    def logpdf(self) -> jax.Array:
        ...

    @abc.abstractmethod
    def __call__(self, sumw: jax.Array) -> jax.Array:
        ...


class FreeFloating(Penalty):
    type: str = "r"

    @property
    def default_scan_points(self) -> jax.Array:
        return jnp.linspace(self.parameter.bounds[0], self.parameter.bounds[1], 100)

    @property
    def logpdf(self) -> jax.Array:
        return jnp.array(0.0)

    def __call__(self, sumw: jax.Array) -> jax.Array:
        return self.parameter.value * sumw


class Normal(Penalty):
    type: str = "gauss"

    width: jax.Array = eqx.field(converter=jax.numpy.asarray)
    logpdf_maximum: jax.Array = eqx.field(converter=jax.numpy.asarray)

    def __init__(self, parameter: Parameter, width: jax.Array) -> None:
        super().__init__(parameter=parameter)
        self.width = width
        self.logpdf_maximum = jax.scipy.stats.norm.logpdf(0, loc=0, scale=self.width)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameter={self.parameter}, width={self.width})"

    @property
    def default_scan_points(self) -> jax.Array:
        return jnp.linspace(self.value - 7 * self.width, self.parameter.value + 7 * self.width, 100)

    @property
    def logpdf(self) -> jax.Array:
        unnormalized = jax.scipy.stats.norm.logpdf(self.parameter.value, loc=0, scale=self.width)
        return unnormalized - self.logpdf_maximum

    def __call__(self, sumw: jax.Array) -> jax.Array:
        return (self.parameter.value + 1) * sumw


class Shape(Penalty):
    type: str = "shape"

    up: jax.Array = eqx.field(converter=jax.numpy.asarray)
    down: jax.Array = eqx.field(converter=jax.numpy.asarray)
    logpdf_maximum: jax.Array = eqx.field(converter=jax.numpy.asarray)

    def __init__(
        self,
        parameter: Parameter,
        up: jax.Array,
        down: jax.Array,
    ) -> None:
        super().__init__(parameter=parameter)
        self.up = up  # +1 sigma
        self.down = down  # -1 sigma
        self.logpdf_maximum = jax.scipy.stats.norm.logpdf(0, loc=0, scale=self.width)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameter={self.parameter}, width={self.width})"

    @property
    def width(self):
        return 1.0

    @property
    def default_scan_points(self) -> jax.Array:
        return jnp.linspace(
            self.parameter.value - 7 * self.width, self.parameter.value + 7 * self.width, 100
        )

    @eqx.filter_jit
    def vshift(self, sumw: jax.Array) -> jax.Array:
        factor = self.parameter.value + 1
        dx_sum = self.up + self.down - 2 * sumw
        dx_diff = self.up - self.down

        # taken from https://github.com/nsmith-/jaxfit/blob/8479cd73e733ba35462287753fab44c0c560037b/src/jaxfit/roofit/combine.py#L173C6-L192
        _asym_poly = jnp.array([3.0, -10.0, 15.0, 0.0]) / 8.0

        abs_value = jnp.abs(factor)
        morph = 0.5 * (
            dx_diff * factor
            + dx_sum
            * jnp.where(
                abs_value > 1.0,
                abs_value,
                jnp.polyval(_asym_poly, factor * factor),
            )
        )

        return morph

    @property
    def logpdf(self) -> jax.Array:
        unnormalized = jax.scipy.stats.norm.logpdf(self.parameter.value, loc=0, scale=self.width)
        return unnormalized - self.logpdf_maximum

    def __call__(self, sumw: jax.Array) -> jax.Array:
        return jax.numpy.clip(sumw + self.vshift(sumw=sumw), a_min=0.0)


class LogNormal(Penalty):
    type: str = "lnN"

    width: jax.Array | tuple[jax.Array, jax.Array]
    logpdf_maximum: jax.Array = eqx.field(converter=jax.numpy.asarray)

    def __init__(
        self,
        parameter: Parameter,
        width: jax.Array | tuple[jax.Array, jax.Array],
    ) -> None:
        super().__init__(parameter=parameter)
        self.width = width
        self.logpdf_maximum = jax.scipy.stats.norm.logpdf(0, loc=0, scale=self.scale)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameter={self.parameter}, width={self.width})"

    @property
    def default_scan_points(self) -> jax.Array:
        if isinstance(self.width, tuple):
            down, up = self.width
        else:
            down = up = self.width
        return jnp.linspace(self.parameter.value - 7 * down, self.parameter.value + 7 * up, 100)

    @property
    def scale(self) -> jax.Array:
        if isinstance(self.width, tuple):
            down, up = self.width
            scale = jnp.where(self.parameter.value > 0, up, down)
        else:
            scale = self.width
        return scale

    @property
    def logpdf(self) -> jax.Array:
        unnormalized = jax.scipy.stats.norm.logpdf(self.parameter.value, loc=0, scale=self.scale)
        return unnormalized - self.logpdf_maximum

    def __call__(self, sumw: jax.Array) -> jax.Array:
        return jnp.exp(self.parameter.value) * sumw


class Poisson(Penalty):
    type = "poisson"

    rate: jax.Array = eqx.field(converter=jax.numpy.asarray)
    logpdf_maximum: jax.Array = eqx.field(converter=jax.numpy.asarray)

    def __init__(
        self,
        parameter: Parameter,
        rate: jax.Array,
    ) -> None:
        super().__init__(parameter=parameter)
        self.rate = rate
        self.logpdf_maximum = jax.scipy.stats.poisson.logpmf(self.rate, self.rate)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(parameter={self.parameter}, lambda={self.rate})"

    @property
    def default_scan_points(self) -> jax.Array:
        return jnp.linspace(
            self.parameter.value - 7 * jnp.sqrt(self.rate),
            self.parameter.value + 7 * jnp.sqrt(self.rate),
            100,
        )

    @property
    def logpdf(self) -> jax.Array:
        unnormalized = jax.scipy.stats.poisson.logpmf(
            self.rate, (self.parameter.value + 1) * self.rate
        )
        return unnormalized - self.logpdf_maximum

    def __call__(self, sumw: jax.Array) -> jax.Array:
        return (self.parameter.value + 1) * sumw
