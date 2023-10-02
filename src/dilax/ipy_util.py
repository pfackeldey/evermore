from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from dilax.model import Model


def interactive(model: Model) -> None:
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    def slider(v: float | jax.Array) -> widgets.FloatSlider:
        return widgets.FloatSlider(min=v - 2, max=v + 2, step=0.01, value=v)

    fig, ax = plt.subplots()

    expectation = model.evaluate().expectation()
    bins = jnp.arange(expectation.size)

    art = ax.bar(bins, expectation, color="gray")

    @widgets.interact(
        **{name: slider(param.value) for name, param in model.parameters.items()}
    )
    def update(**kwargs: Any) -> None:
        m = model.update(values=kwargs)
        res = m.evaluate()

        expectation = res.expectation()
        print("Expectation:", expectation)
        print("Constraint (logpdf):", m.parameter_constraints())

        nonlocal art
        art.remove()

        art = ax.bar(bins, expectation, color="gray")

    ax.set_xticks(bins)
    ax.set_xticklabels(list(map(str, bins)))
    ax.set_xlabel(r"Bin #")
    ax.set_ylabel(r"S+B model")
    plt.tight_layout()
    plt.show()
