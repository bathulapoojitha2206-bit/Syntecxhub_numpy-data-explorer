"""Unit tests for reshaping, array manipulation, and broadcasting."""

import numpy as np
import pytest
from src.reshaping import (
    validate_reshape,
    execute_reshape,
    execute_flatten_vs_ravel,
    execute_transpose,
    execute_join_operation,
    execute_split_operation,
    execute_expand_or_squeeze,
    execute_repeat_or_tile,
    execute_sort_or_unique,
)
from src.broadcasting import (
    analyze_broadcasting,
    broadcast_and_operate,
    create_broadcasting_scenarios,
)


# ==========================================
# Reshaping & Manipulation Tests
# ==========================================

def test_validate_reshape():
    valid, _ = validate_reshape(12, (3, 4))
    assert valid is True

    valid_neg, msg = validate_reshape(12, (2, -1))
    assert valid_neg is True
    assert "inferred as 6" in msg

    invalid_size, _ = validate_reshape(12, (3, 5))
    assert invalid_size is False

    invalid_two_negs, _ = validate_reshape(12, (-1, -1))
    assert invalid_two_negs is False


def test_execute_reshape():
    arr = np.arange(12)
    reshaped, is_view, code = execute_reshape(arr, (3, 4))
    assert reshaped.shape == (3, 4)
    assert is_view is True

    with pytest.raises(ValueError):
        execute_reshape(arr, (5, 5))


def test_flatten_vs_ravel():
    arr = np.arange(6).reshape(2, 3)

    ravel_arr, is_view_r, _, note_r = execute_flatten_vs_ravel(arr, "ravel")
    assert ravel_arr.shape == (6,)
    assert is_view_r is True
    assert "VIEW" in note_r

    flat_arr, is_view_f, _, note_f = execute_flatten_vs_ravel(arr, "flatten")
    assert flat_arr.shape == (6,)
    assert is_view_f is False
    assert "COPY" in note_f


def test_transpose():
    arr = np.arange(6).reshape(2, 3)
    trans, is_view, code = execute_transpose(arr)
    assert trans.shape == (3, 2)
    assert is_view is True


def test_join_operations():
    a = np.ones((2, 2))
    b = np.zeros((2, 2))

    concat = execute_join_operation(a, b, "concatenate", axis=0)[0]
    assert concat.shape == (4, 2)

    vstacked = execute_join_operation(a, b, "vstack")[0]
    assert vstacked.shape == (4, 2)

    hstacked = execute_join_operation(a, b, "hstack")[0]
    assert hstacked.shape == (2, 4)

    stacked = execute_join_operation(a, b, "stack", axis=0)[0]
    assert stacked.shape == (2, 2, 2)


def test_split_operations():
    arr = np.arange(10)

    # Equal split
    splits, _ = execute_split_operation(arr, sections=5, axis=0, allow_unequal=False)
    assert len(splits) == 5
    assert splits[0].shape == (2,)

    # Unequal split with regular split should raise ValueError
    with pytest.raises(ValueError, match="does not result in an equal division"):
        execute_split_operation(arr, sections=3, axis=0, allow_unequal=False)

    # Unequal split with array_split should succeed
    splits_unequal, _ = execute_split_operation(arr, sections=3, axis=0, allow_unequal=True)
    assert len(splits_unequal) == 3


def test_expand_squeeze_repeat_tile():
    arr = np.array([1, 2, 3])
    expanded, _, _ = execute_expand_or_squeeze(arr, "expand_dims", axis=0)
    assert expanded.shape == (1, 3)

    squeezed, _, _ = execute_expand_or_squeeze(expanded, "squeeze")
    assert squeezed.shape == (3,)

    repeated, _ = execute_repeat_or_tile(arr, "repeat", repeats=2)
    assert np.array_equal(repeated, [1, 1, 2, 2, 3, 3])

    tiled, _ = execute_repeat_or_tile(arr, "tile", repeats=2)
    assert np.array_equal(tiled, [1, 2, 3, 1, 2, 3])


def test_sort_and_unique():
    arr = np.array([4, 2, 5, 2, 1, 4])
    sorted_arr, _, _ = execute_sort_or_unique(arr, "sort")
    assert np.array_equal(sorted_arr, [1, 2, 2, 4, 4, 5])

    uniques, counts, _ = execute_sort_or_unique(arr, "unique")
    assert np.array_equal(uniques, [1, 2, 4, 5])
    assert np.array_equal(counts, [1, 2, 2, 1])


# ==========================================
# Broadcasting Tests
# ==========================================

def test_analyze_broadcasting_compatible():
    # (3, 4) with (4,) -> compatible, result (3, 4)
    res = analyze_broadcasting((3, 4), (4,))
    assert res["is_compatible"] is True
    assert res["result_shape"] == (3, 4)
    assert res["padded_b"] == (1, 4)

    # (3, 1) with (1, 4) -> (3, 4)
    res_outer = analyze_broadcasting((3, 1), (1, 4))
    assert res_outer["is_compatible"] is True
    assert res_outer["result_shape"] == (3, 4)


def test_analyze_broadcasting_incompatible():
    # (2, 3) with (2,) -> incompatible!
    res = analyze_broadcasting((2, 3), (2,))
    assert res["is_compatible"] is False
    assert res["result_shape"] is None
    assert "CONFLICT" in res["steps"][-1]["explanation"]


def test_broadcast_and_operate():
    a = np.ones((3, 4))
    b = np.array([1, 2, 3, 4])
    res, shape, code = broadcast_and_operate(a, b, "add")
    assert shape == (3, 4)
    assert np.all(res[0] == [2, 3, 4, 5])

    # Incompatible should raise ValueError
    with pytest.raises(ValueError, match="could not be broadcast together"):
        broadcast_and_operate(np.ones((2, 3)), np.ones((2,)))


def test_broadcasting_scenarios():
    scenarios = create_broadcasting_scenarios()
    assert "1. Scalar & 2D Matrix" in scenarios
    assert "3. Column Vector & Row Vector (Outer Expansion)" in scenarios
