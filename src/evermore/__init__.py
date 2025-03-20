"""
evermore: Differentiable (binned) likelihoods in JAX.
"""

from __future__ import annotations

__author__ = "Peter Fackeldey"
__email__ = "peter.fackeldey@rwth-aachen.de"
__copyright__ = "Copyright 2023, Peter Fackeldey"
__credits__ = ["Peter Fackeldey"]
__contact__ = "https://github.com/pfackeldey/evermore"
__license__ = "BSD-3-Clause"
__status__ = "Development"
__version__ = "0.2.10"


# expose public API

__all__ = [
    "Modifier",
    "NormalParameter",
    # explicitly expose some classes
    "Parameter",
    "__version__",
    "effect",
    "loss",
    "modifier",
    "parameter",
    "pdf",
    "staterror",
    "util",
    "visualization",
]


def __dir__():
    return __all__


from evermore import (  # noqa: E402
    effect,
    loss,
    modifier,
    parameter,
    pdf,
    staterror,
    util,
    visualization,
)
from evermore.modifier import Modifier  # noqa: E402
from evermore.parameter import (  # noqa: E402,
    NormalParameter,
    Parameter,
)
