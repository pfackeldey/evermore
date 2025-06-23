from __future__ import annotations

import equinox as eqx
from model import params

if __name__ == "__main__":
    # see: https://docs.kidger.site/equinox/api/serialisation/
    # save parameters
    eqx.tree_serialise_leaves("some_filename.eqx", params)

    # load parameters back into params
    params_loaded = eqx.tree_deserialise_leaves("some_filename.eqx", params)
