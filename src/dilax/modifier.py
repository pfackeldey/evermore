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
from dilax.parameter import Parameter

if TYPE_CHECKING:
    from dilax.effect import Effect

__all__ = [
    "modifier",
    "compose",
    "staterror",
    "autostaterrors",
]


def __dir__():
    return __all__


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
    parameters: dict[str, Parameter]
    sumw: jax.Array
    sumw2: jax.Array
    sumw2sqrt: jax.Array
    widths: jax.Array
    mask: jax.Array
    threshold: float

    def __init__(
        self,
        parameters: dict[str, Parameter],
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

        for i, name in enumerate(self.parameters):
            param = self.parameters[name]
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
        if not self.threshold > 0.0:
            msg = f"Threshold must be >= 0.0, got: {self.threshold}"
            raise ValueError(msg)

    def scale_factor(self, sumw: jax.Array) -> jax.Array:
        from functools import partial

        assert len(sumw) == len(self.parameters) == len(self.sumw2)

        values = jnp.concatenate([param.value for param in self.parameters.values()])
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

        # if all true in mask use poisson
        if jnp.all(self.mask):
            return jax.vmap(_poisson_mod)(values, _widths, idxs)

        # if all false in mask use gauss
        if jnp.all(~self.mask):
            return jax.vmap(_gauss_mod)(values, _widths, idxs)

        # if mixed use jnp.where
        return jnp.where(
            self.mask,
            jax.vmap(_poisson_mod)(values, _widths, idxs),
            jax.vmap(_gauss_mod)(values, _widths, idxs),
        )


class autostaterrors(eqx.Module):
    class Mode(eqx.Enumeration):
        poisson = "Poisson per process and bin"
        poisson_gauss = "Poisson (Gauss) per process and bin if sumw < (>) threshold"
        barlow_beeston_lite = "Barlow-Beeston (lite) approach"

    sumw: dict[str, jax.Array]
    sumw2: dict[str, jax.Array]
    masks: dict[str, jax.Array]
    threshold: float = 10.0
    mode: str = Mode.barlow_beeston_lite
    key_template: str = "__staterror_{process}__"

    def __init__(
        self,
        sumw: dict[str, jax.Array],
        sumw2: dict[str, jax.Array],
        threshold: float = 10.0,
        mode: str = Mode.barlow_beeston_lite,
        key_template: str = "__staterror_{process}__",
    ) -> None:
        self.sumw = sumw
        self.sumw2 = sumw2
        self.masks = {p: _sumw < threshold for p, _sumw in sumw.items()}
        self.threshold = threshold
        self.mode = mode
        self.key_template = key_template

    def __check_init__(self):
        if jax.tree_util.tree_structure(self.sumw) != jax.tree_util.tree_structure(
            self.sumw2
        ):  # type: ignore[operator]
            msg = (
                "The structure of `sumw` and `sumw2` needs to be identical, got "
                f"`sumw`: {jax.tree_util.tree_structure(self.sumw)}) and "
                f"`sumw2`: {jax.tree_util.tree_structure(self.sumw2)})"
            )
            raise ValueError(msg)
        if not self.threshold > 0.0:
            msg = f"Threshold must be >= 0.0, got: {self.threshold}"
            raise ValueError(msg)
        if not isinstance(self.mode, self.Mode):
            msg = f"Mode must be of type {self.Mode}, got: {self.mode}"
            raise ValueError(msg)

    def prepare(
        self
    ) -> tuple[dict[str, dict[str, Parameter]], dict[str, dict[str, eqx.Partial]]]:
        """
        Helper to automatically create parameters used by `staterror`
        for the initialisation of a `dlx.Model`.

        _Caution_: This function is not compatible with JAX-transformations (e.g. `jax.jit`)!

        Example:

            .. code-block:: python

                import jax.numpy as jnp
                import dilax as dlx

                sumw = {
                    "signal": jnp.array([5, 20, 30]),
                    "background": jnp.array([5, 20, 30]),
                }

                sumw2 = {
                    "signal": jnp.array([5, 20, 30]),
                    "background": jnp.array([5, 20, 30]),
                }


                auto = dlx.autostaterrors(
                    sumw=sumw,
                    sumw2=sumw2,
                    threshold=10.0,
                    mode=dlx.autostaterrors.Mode.poisson,
                )
                parameters, staterrors = auto.prepare()

                # barlow-beeston-lite
                auto2 = dlx.autostaterrors(
                    sumw=sumw,
                    sumw2=sumw2,
                    threshold=10.0,
                    mode=dlx.autostaterrors.Mode.barlow_beeston_lite,
                )
                parameters2, staterrors2 = auto2.prepare()

                # materialize:
                process = "signal"
                pkey = auto.key_template.format(process=process)
                modify = staterrors[pkey](parameters[pkey])
                modified_process = modify(sumw[process])
        """
        import equinox as eqx

        parameters: dict[str, dict[str, Parameter]] = {}
        staterrors: dict[str, dict[str, eqx.Partial]] = {}
        for process, _sumw in self.sumw.items():
            key = self.key_template.format(process=process)
            process_parameters = parameters[key] = {}
            mask = self.masks[process]
            for i in range(len(_sumw)):
                pkey = f"{process}_{i}"
                if self.mode == self.Mode.barlow_beeston_lite and not mask[i]:
                    # we merge all processes into one parameter
                    # for the barlow-beeston-lite approach where
                    # the bin content is above a certain threshold
                    pkey = f"{i}"
                process_parameters[pkey] = Parameter(value=jnp.array(0.0))
            # prepare staterror
            kwargs = {
                "sumw": _sumw,
                "sumw2": self.sumw2[process],
                "threshold": self.threshold,
            }
            if self.mode == self.Mode.poisson:
                kwargs["threshold"] = jnp.inf  # inf -> always poisson
            elif self.mode == self.Mode.barlow_beeston_lite:
                kwargs["sumw"] = jnp.where(
                    mask, _sumw, sum(jax.tree_util.tree_leaves(self.sumw))
                )
                kwargs["sumw2"] = jnp.where(
                    mask,
                    self.sumw2[process],
                    sum(jax.tree_util.tree_leaves(self.sumw2)),
                )
            staterrors[key] = eqx.Partial(staterror, **kwargs)
        return parameters, staterrors
