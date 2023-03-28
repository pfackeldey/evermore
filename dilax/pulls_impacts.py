from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
from tqdm.auto import tqdm

from dilax.likelihood import nll
from dilax.model import Model
from dilax.optimizer import JaxOptimizer


@partial(jax.jit, static_argnames=("param_name", "fit", "optimizer"))
def profile_parameter(
    param_name: str,
    scan_points: jax.Array,
    model: Model,
    observation: jax.Array,
    fit: bool,
    optimizer: JaxOptimizer,
) -> jax.Array:
    @partial(jax.jit, static_argnames=("param_name", "fit", "optimizer"))
    def _fixed_poi_fit(
        param_name: str,
        scan_point: jax.Array,
        model: Model,
        observation: jax.Array,
        fit: bool,
        optimizer: JaxOptimizer,
    ) -> jax.Array:
        # calculate nll for different value of fixed theta
        # fix theta into the model
        model = model.apply(parameters={param_name: scan_point})
        init_params = model.parameter_strengths
        init_params.pop(param_name, 1)
        # calculate nll for the different value of mu
        # minimize
        if fit:
            params, state = optimizer.fit(
                fun=nll,
                init_params=init_params,
                model=model,
                observation=observation,
            )
        else:
            params = init_params
        return nll(parameters=params, model=model, observation=observation)

    _fixed_poi_fit_vec = jax.vmap(_fixed_poi_fit, in_axes=(None, 0, None, None, None, None))

    return _fixed_poi_fit_vec(param_name, scan_points, model, observation, fit, optimizer)


@jax.jit
def nll_crossing_idx_from_min(
    scan_points: jax.Array,
    crossing: jax.Array,
    direction: jax.Array,  # -1: left, +1: right
) -> jax.Array:
    start_idx = jnp.argmin(scan_points)

    def body_fn(idx: int) -> int:
        return idx + direction

    def cond_fn(idx: int) -> bool:
        return (scan_points[idx] < crossing) & (idx < scan_points.shape[0] - 1) & (idx > 0)

    idx = jax.lax.while_loop(cond_fn, body_fn, start_idx)
    return jnp.sort(jnp.array([idx - 1, idx + 1]))


def pulls(
    poi: str,
    model: Model,
    observation: jax.Array,
    parameters: None | dict[str, jax.Array],
    optimizer: JaxOptimizer,
) -> jax.Array:
    # one global fit
    init_params = {name: param.strength for name, param in model.parameters.items()}
    best_fit_params, state = optimizer.fit(
        fun=nll, init_params=init_params, model=model, observation=observation
    )

    # calculate nll for different value of fixed parameters
    # fix parameters into the model
    if parameters is None:
        parameters = {
            key: parameter.default_scan_points for key, parameter in model.parameters.items()
        }
        # remove poi from parameters
        parameters.pop(poi, 1)
    pulls = {}

    for key, scan_range in (pbar := tqdm(parameters.items())):
        pbar.set_description(f"Calculate pull of '{key}'")
        scan_points = profile_parameter(
            param_name=key,
            scan_points=scan_range,
            model=model,
            observation=observation,
            fit=False,  # prefit uncertainty
            optimizer=optimizer,
        )
        mask = jnp.isfinite(scan_points)
        scan_points = scan_points[mask]
        scan_points -= scan_points.min()
        # interpolation
        xvals = scan_range[mask]
        yvals = scan_points

        def interpolation(x: jax.Array, offset: jax.Array = jnp.array(0.0)) -> jax.Array:
            return jnp.interp(x, xvals, yvals - offset)

        # 1 sigma uncertainty
        nom_pull = best_fit_params[key] - init_params[key]
        sigma = jnp.array(1.0)

        direction = jnp.sign(nom_pull).astype(int)
        if direction == 0:
            pulls[key] = nom_pull
        else:
            assert direction in [-1, 1]
            # norm by 1 sigma uncertainty
            idx = nll_crossing_idx_from_min(scan_points, sigma, direction)
            lower, upper = xvals[idx]
            interp = partial(interpolation, offset=sigma)
            # important that the sign of the interpolation changes
            assert interp(lower) * interp(upper) < 0

            unc = (
                jaxopt.Bisection(
                    optimality_fun=interp,
                    lower=lower,
                    upper=upper,
                )
                .run()
                .params
            )

            pulls[key] = nom_pull / unc
    return pulls
