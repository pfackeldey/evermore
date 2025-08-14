import optimistix as optx
import wadler_lindig as wl
from model import hists, loss, observation, params

import evermore as evm


def fit(params, hists, observation):
    solver = optx.BFGS(rtol=1e-5, atol=1e-7)

    dynamic, static = evm.tree.partition(params)

    def optx_loss(dynamic, args):
        return loss(dynamic, *args)

    fitresult = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, hists, observation),
        options={},
        max_steps=10_000,
        throw=True,
    )
    return evm.tree.combine(fitresult.value, static)


if __name__ == "__main__":
    bestfit_params = fit(params, hists, observation)

    print("Bestfit parameter:")
    wl.pprint(evm.tree.pure(bestfit_params), short_arrays=False)
