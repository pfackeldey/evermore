"""
evermore: Differentiable (binned) likelihoods in JAX.
"""

from __future__ import annotations

__author__ = "Peter Fackeldey"
__version__ = "0.4.1"


# expose public API

__all__ = [
    "BaseParameter",
    "Modifier",
    "NormalParameter",
    # explicitly expose some classes
    "Parameter",
    "V",
    "__version__",
    "effect",
    "filter",
    "loss",
    "modifier",
    "parameter",
    "pdf",
    "sample",
    "staterror",
    "transform",
    "util",
]


def __dir__():
    return __all__


from evermore import loss, pdf, util  # ruff:ignore[module-import-not-at-top-of-file]
from evermore.binned import (  # ruff:ignore[module-import-not-at-top-of-file]
    effect,
    modifier,
    staterror,
)
from evermore.binned.modifier import (  # ruff:ignore[module-import-not-at-top-of-file]
    Modifier,
)
from evermore.parameters import (  # ruff:ignore[module-import-not-at-top-of-file]
    filter,
    parameter,
    sample,
    transform,
)
from evermore.parameters.parameter import (  # ruff:ignore[module-import-not-at-top-of-file]
    BaseParameter,
    NormalParameter,
    Parameter,
    V,
)
