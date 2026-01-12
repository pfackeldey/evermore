import optimistix as optx
import wadler_lindig as wl
from flax import nnx
from model import hists, loss, make_model, observation

import evermore as evm


def fit(model, hists, observation):
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    # split the model into metadata (graphdef), dynamic and static part
    graphdef, dynamic, static = nnx.split(model, evm.filter.is_dynamic_parameter, ...)

    # run the minimization
    fitresult = optx.minimise(
        loss,
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
    model = make_model()
    bestfit_params = fit(model, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(nnx.pure(bestfit_params), short_arrays=False)
