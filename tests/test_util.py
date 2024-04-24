from __future__ import annotations

from evermore.util import sum_over_leaves


def test_sum_over_leaves():
    tree = {"a": 1, "b": {"c": 2, "d": 3}}
    assert sum_over_leaves(tree) == 6
