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
__version__ = "0.1.6"


# expose public API

__all__ = [
    "__version__",
    "effect",
    "ipy_util",
    "likelihood",
    "optimizer",
    "pdf",
    "util",
    # explicitely expose some classes
    "Model",
    "Result",
    "Parameter",
    "modifier",
    "staterror",
    "autostaterrors",
    "compose",
]


def __dir__():
    return __all__


from dilax import (  # noqa: E402
    effect,
    ipy_util,
    likelihood,
    optimizer,
    pdf,
    util,
)
from dilax.model import Model, Result  # noqa: E402
from dilax.modifier import (  # noqa: E402
    autostaterrors,
    compose,
    modifier,
    staterror,
)
from dilax.parameter import Parameter  # noqa: E402
