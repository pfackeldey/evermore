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
__version__ = "0.2.2"


# expose public API

__all__ = [
    "__version__",
    "effect",
    "loss",
    "parameter",
    "pdf",
    "util",
    "sample",
    "modifier",
    # explicitely expose some classes
    "Parameter",
    "Modifier",
    "ModifierBase",
]


def __dir__():
    return __all__


from evermore import (  # noqa: E402
    effect,
    loss,
    modifier,
    parameter,
    pdf,
    sample,
    util,
)
from evermore.modifier import Modifier, ModifierBase  # noqa: E402
from evermore.parameter import Parameter  # noqa: E402
