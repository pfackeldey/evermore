from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from evermore.util import sum_over_leaves, tree_stack

jax.config.update("jax_enable_x64", True)


def test_sum_over_leaves():
    tree = {"a": 1, "b": {"c": 2, "d": 3}}
    assert sum_over_leaves(tree) == 6


def test_tree_stack_stacks_arrays_with_batch_axis():
    trees = [
        {"a": jnp.array([1.0]), "b": jnp.array([2.0])},
        {"a": jnp.array([3.0]), "b": jnp.array([4.0])},
    ]
    stacked = tree_stack(trees)

    assert stacked["a"].shape == (2, 1)
    assert stacked["b"].shape == (2, 1)


def test_tree_stack_supports_broadcasting_scalars():
    tree_one = {"a": jnp.array([1.0])}
    tree_two = {"a": jnp.array(2.0)}

    stacked = tree_stack([tree_one, tree_two], broadcast_leaves=True)
    assert stacked["a"].shape == (2, 1)


def test_tree_stack_raises_for_mismatched_structure():
    trees = [
        {"a": jnp.array([1.0])},
        {"a": jnp.array([2.0]), "b": jnp.array([3.0])},
    ]

    with pytest.raises(
        ValueError, match="All static trees must have the same structure"
    ):
        tree_stack(trees)
