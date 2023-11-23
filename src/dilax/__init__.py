"""
dilax: Differentiable (binned) likelihoods in JAX.
"""
from __future__ import annotations

__author__ = "Peter Fackeldey"
__email__ = "peter.fackeldey@rwth-aachen.de"
__copyright__ = "Copyright 2023, Peter Fackeldey"
__credits__ = ["Peter Fackeldey"]
__contact__ = "https://github.com/pfackeldey/dilax"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "0.1.5"


# expose public API

__all__ = [
    "ipy_util",
    "likelihood",
    "model",
    "optimizer",
    "parameter",
    # "pdf", # this should not be needed in public API
    "util",
    "__version__",
]


def __dir__():
    return __all__


from dilax import (  # noqa: E402
    ipy_util,
    likelihood,
    model,
    optimizer,
    parameter,
    # pdf,  # this should not be needed in public API
    util,
)
