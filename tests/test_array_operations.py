"""Unit tests for indexing, slicing, math, and statistical operations."""

import numpy as np
import pytest
from src.indexing_slicing import (
    safe_index_1d,
    safe_index_2d,
    slice_array,
    slice_2d_submatrix,
    select_row_or_col,
    filter_boolean,
    fancy_indexing,
    reverse_array,
    extract_diagonal,
    demonstrate_view_vs_copy,
)
from src.mathematical_operations import (
    compute_arithmetic,
    compute_unary_math,
    compute_comparison,
    compute_where,
    compute_elementwise_min_max,
)
from src.statistical_operations import (
    compute_all_statistics,
    compute_cumulative_stats,
    get_axis_explanation,
    get_ddof_explanation,
    generate_sample_datasets,
)


# ==========================================
# Indexing & Slicing Tests
# ==========================================

def test_safe_indexing():
    arr1d = np.array([10, 20, 30, 40])
    val, code = safe_index_1d(arr1d, 2)
    assert val == 30
    assert "arr[2]" in code

    val_neg, _ = safe_index_1d(arr1d, -1)
    assert val_neg == 40

    with pytest.raises(IndexError):
        safe_index_1d(arr1d, 10)

    arr2d = np.array([[1, 2], [3, 4]])
    val2, _ = safe_index_2d(arr2d, 1, 0)
    assert val2 == 3

    with pytest.raises(IndexError):
        safe_index_2d(arr2d, 5, 0)


def test_slice_array():
    arr = np.array([0, 1, 2, 3, 4, 5])
    slc, code = slice_array(arr, 1, 5, 2)
    assert np.array_equal(slc, [1, 3])
    assert "arr[1:5:2]" in code

    with pytest.raises(ValueError, match="step cannot be zero"):
        slice_array(arr, 1, 5, 0)


def test_slice_2d_submatrix():
    arr = np.arange(16).reshape(4, 4)
    sub, code = slice_2d_submatrix(arr, 1, 3, 1, 1, 4, 1)
    assert sub.shape == (2, 3)
    assert np.array_equal(sub, [[5, 6, 7], [9, 10, 11]])


def test_select_row_or_col():
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    row, code_r = select_row_or_col(arr, 1, axis=0)
    assert np.array_equal(row, [4, 5, 6])

    col, code_c = select_row_or_col(arr, 2, axis=1)
    assert np.array_equal(col, [3, 6])


def test_filter_boolean():
    arr = np.array([1, 5, 10, 15, 20])
    filtered, mask, code = filter_boolean(arr, ">", 10)
    assert np.array_equal(filtered, [15, 20])
    assert np.array_equal(mask, [False, False, False, True, True])

    evens, _, _ = filter_boolean(arr, "even", 0)
    assert np.array_equal(evens, [10, 20])


def test_fancy_indexing():
    arr = np.array([10, 20, 30, 40, 50])
    res, _ = fancy_indexing(arr, [0, 3, 1])
    assert np.array_equal(res, [10, 40, 20])


def test_reverse_and_diagonal():
    arr = np.array([[1, 2], [3, 4]])
    rev, _ = reverse_array(arr)
    assert np.array_equal(rev, [[4, 3], [2, 1]])

    diag, _ = extract_diagonal(arr, offset=0)
    assert np.array_equal(diag, [1, 4])


def test_view_vs_copy_mutation():
    base = np.array([10.0, 20.0, 30.0, 40.0])
    demo = demonstrate_view_vs_copy(base, slice_start=0, slice_stop=2, mutation_value=999.0)

    # In view experiment, original array must be mutated
    assert demo["view_shares_memory"] is True
    assert demo["view_modified_original"][0] == 999.0
    assert demo["view_modified_original"][1] == 999.0

    # In copy experiment, original array must NOT be mutated
    assert demo["copy_modified_original"][0] == 10.0
    assert demo["copy_modified_original"][1] == 20.0


# ==========================================
# Mathematical Operations Tests
# ==========================================

def test_compute_arithmetic():
    a = np.array([10.0, 20.0, 30.0])
    b = np.array([2.0, 5.0, 10.0])

    res_add, _, _, _ = compute_arithmetic(a, b, "add")
    assert np.allclose(res_add, [12.0, 25.0, 40.0])

    res_div, _, _, warn = compute_arithmetic(a, b, "divide")
    assert np.allclose(res_div, [5.0, 4.0, 3.0])
    assert warn is None

    # Test division by zero
    b_zero = np.array([2.0, 0.0, 10.0])
    res_zero, _, _, warn_zero = compute_arithmetic(a, b_zero, "divide")
    assert np.isinf(res_zero[1])
    assert warn_zero is not None


def test_compute_unary_math():
    a = np.array([4.0, 9.0, 16.0])
    res_sqrt, _, _, warn = compute_unary_math(a, "sqrt")
    assert np.allclose(res_sqrt, [2.0, 3.0, 4.0])
    assert warn is None

    # Negative sqrt
    a_neg = np.array([-4.0, 9.0])
    res_neg, _, _, warn_neg = compute_unary_math(a_neg, "sqrt")
    assert np.isnan(res_neg[0])
    assert warn_neg is not None

    # Logarithm
    a_log = np.array([np.e, 1.0])
    res_log, _, _, _ = compute_unary_math(a_log, "log")
    assert np.allclose(res_log, [1.0, 0.0])


def test_compute_comparison_and_where():
    a = np.array([1, 5, 10])
    comp, _, _ = compute_comparison(a, 5, ">")
    assert np.array_equal(comp, [False, False, True])

    where_res, _, _ = compute_where(comp, 100, 0)
    assert np.array_equal(where_res, [0, 0, 100])

    max_res, _, _ = compute_elementwise_min_max(a, 4, mode="max")
    assert np.array_equal(max_res, [4, 5, 10])


# ==========================================
# Statistical Operations Tests
# ==========================================

def test_compute_all_statistics():
    mat = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0]
    ])

    # axis=None
    stats_global = compute_all_statistics(mat, axis=None, ddof=0)
    assert stats_global["mean"] == 35.0
    assert stats_global["sum"] == 210.0
    assert stats_global["min"] == 10.0
    assert stats_global["max"] == 60.0

    # axis=0 (down the rows -> columns)
    stats_axis0 = compute_all_statistics(mat, axis=0, ddof=0)
    assert np.allclose(stats_axis0["mean"], [25.0, 35.0, 45.0])
    assert np.allclose(stats_axis0["sum"], [50.0, 70.0, 90.0])

    # axis=1 (across columns -> rows)
    stats_axis1 = compute_all_statistics(mat, axis=1, ddof=0)
    assert np.allclose(stats_axis1["mean"], [20.0, 50.0])


def test_ddof_distinction():
    arr = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    var_pop = np.var(arr, ddof=0)
    var_sample = np.var(arr, ddof=1)
    assert var_sample > var_pop
    assert var_sample == pytest.approx(var_pop * len(arr) / (len(arr) - 1))


def test_sample_datasets():
    datasets = generate_sample_datasets()
    assert "Student Marks (15 x 5)" in datasets
    df, arr, desc = datasets["Student Marks (15 x 5)"]
    assert arr.shape == (15, 5)
    assert len(df) == 15
    assert len(desc) > 0
