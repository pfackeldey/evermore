# Parameter PyTree Management

evermore now delegates all parameter PyTree manipulation to the utilities
provided by [`flax.nnx`](https://flax.readthedocs.io/en/latest/nnx/index.html).
The previous `evermore.parameters.tree` module has been removed in favour of
directly composing `nnx` primitives together with the filters exposed in
`evermore.parameters.filter`.

```{important}
If you migrated from an older release, replace calls to
`evm.tree.partition`, `evm.tree.combine`, or `evm.tree.only` with the
corresponding `flax.nnx` helpers (`nnx.split`, `nnx.merge`, `nnx.state`, and
`.filter`) together with `evermore.filter` predicates.
```

Typical workflows make heavy use of `nnx.split`, `nnx.merge`, and `nnx.state`:

```{code-block} python
from flax import nnx
import evermore as evm


params = {
    "mu": evm.Parameter(name="signal strength"),
    "theta": evm.NormalParameter(tags=frozenset({"theory"})),
}

# extract only the parameter state for further processing
params_state, _ = nnx.state(params, evm.filter.is_parameter, ...)

# split into differentiable vs. static parameters
graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)

# merge updated values back into the full PyTree
updated = nnx.merge(graphdef, dynamic, static, copy=True)
```

See also:

- {mod}`evermore.parameters.filter`
- {mod}`evermore.parameters.sample`
- {mod}`evermore.parameters.transform`
- [`flax.nnx` documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)
