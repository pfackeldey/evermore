import optimistix as optx
import wadler_lindig as wl
from flax import nnx
from model import hists, loss, observation, params

import evermore as evm


def fit(params, hists, observation):
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)

    def optx_loss(dynamic, args):
        graphdef, static, hists, observation = args
        params = nnx.merge(graphdef, dynamic, static)
        return loss(params, hists=hists, observation=observation)

    fitresult = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(graphdef, static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )
    return nnx.merge(graphdef, fitresult.value, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(nnx.pure(bestfit_params), short_arrays=False)
