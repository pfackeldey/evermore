from __future__ import annotations

import jax
from flax import nnx

import evermore as evm
from evermore.parameters import filter as filt

jax.config.update("jax_enable_x64", True)


def _build_params():
    return {
        "free": evm.Parameter(name="free"),
        "frozen": evm.Parameter(name="frozen", frozen=True),
        "tagged": evm.Parameter(name="tagged", tags=frozenset({"theory"})),
        "other": 0.0,
    }


def test_is_dynamic_parameter_splits_out_frozen():
    params = _build_params()

    graphdef, dynamic, static = nnx.split(params, filt.is_dynamic_parameter, ...)

    dynamic_pure = nnx.pure(dynamic)

    assert "free" in dynamic_pure
    assert "frozen" not in dynamic_pure

    merged = nnx.merge(graphdef, dynamic, static)
    assert merged["free"].value == params["free"].value
    assert merged["frozen"].value == params["frozen"].value
    assert merged["other"] == params["other"]


def test_has_name_filter_selects_expected_parameter():
    params_state, _ = nnx.state(_build_params(), filt.is_parameter, ...)
    filtered = params_state.filter(filt.HasName("tagged"))

    assert set(filtered.keys()) == {"tagged"}
    assert filtered["tagged"] == params_state["tagged"]


def test_has_tags_filter_matches_subset():
    params_state, _ = nnx.state(_build_params(), filt.is_parameter, ...)
    filtered = params_state.filter(filt.HasTags(frozenset({"theory"})))

    assert set(filtered.keys()) == {"tagged"}
    assert filtered["tagged"] == params_state["tagged"]
