from __future__ import annotations

import equinox as eqx
from model import model

# see: https://docs.kidger.site/equinox/api/serialisation/

# save processes and parameters
eqx.tree_serialise_leaves("some_filename.eqx", model)

# load processes and parameters back into model
model_loaded = eqx.tree_deserialise_leaves("some_filename.eqx", model)
