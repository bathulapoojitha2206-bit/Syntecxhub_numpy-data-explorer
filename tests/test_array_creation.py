"""Unit tests for array_creation module."""

import numpy as np
import pytest
from src.array_creation import (
    create_from_data,
    create_zeros,
    create_ones,
    create_empty,
    create_arange,
    create_linspace,
    create_eye,
    create_full,
    create_random_array,
    get_array_properties,
    generate_creation_code,
)


def test_create_from_data():
    data = [1, 2, 3, 4]
    arr = create_from_data(data, dtype="float32")
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.shape == (4,)
    assert np.allclose(arr, [1.0, 2.0, 3.0, 4.0])


def test_create_zeros():
    arr = create_zeros((3, 4), dtype=int)
    assert arr.shape == (3, 4)
    assert arr.dtype == int
    assert np.all(arr == 0)


def test_create_ones():
    arr = create_ones((2, 3, 2), dtype=float)
    assert arr.shape == (2, 3, 2)
    assert arr.dtype == float
    assert np.all(arr == 1.0)


def test_create_empty():
    arr, warning_msg = create_empty((3, 3))
    assert arr.shape == (3, 3)
    assert "WARNING" in warning_msg
    assert "uninitialized" in warning_msg.lower()


def test_create_arange():
    arr = create_arange(0, 10, 2)
    assert np.array_equal(arr, [0, 2, 4, 6, 8])

    with pytest.raises(ValueError, match="Step cannot be zero"):
        create_arange(0, 5, 0)


def test_create_linspace():
    arr = create_linspace(0, 10, num=5, endpoint=True)
    assert len(arr) == 5
    assert arr[0] == 0.0
    assert arr[-1] == 10.0
    assert np.allclose(arr, [0.0, 2.5, 5.0, 7.5, 10.0])

    with pytest.raises(ValueError, match="must be positive"):
        create_linspace(0, 10, num=-1)


def test_create_eye():
    eye_mat = create_eye(3)
    assert eye_mat.shape == (3, 3)
    assert np.array_equal(np.diag(eye_mat), [1.0, 1.0, 1.0])
    assert eye_mat.sum() == 3.0

    eye_k = create_eye(3, k=1)
    assert eye_k[0, 1] == 1.0
    assert eye_k[1, 2] == 1.0
    assert eye_k[0, 0] == 0.0

    with pytest.raises(ValueError, match="must be greater than zero"):
        create_eye(0)


def test_create_full():
    arr = create_full((2, 3), fill_value=42)
    assert arr.shape == (2, 3)
    assert np.all(arr == 42)


def test_create_random_array_seed_reproducibility():
    arr1 = create_random_array((4, 4), distribution="uniform", seed=123)
    arr2 = create_random_array((4, 4), distribution="uniform", seed=123)
    assert np.array_equal(arr1, arr2)

    arr_norm = create_random_array((100,), distribution="normal", mean=50.0, std=5.0, seed=42)
    assert arr_norm.shape == (100,)
    assert 40.0 < np.mean(arr_norm) < 60.0

    arr_int = create_random_array((5, 5), distribution="integers", low=10, high=20, seed=42)
    assert arr_int.shape == (5, 5)
    assert np.all(arr_int >= 10) and np.all(arr_int < 20)


def test_get_array_properties():
    arr = np.ones((4, 5), dtype=np.int64)
    props = get_array_properties(arr)
    assert props["shape"] == (4, 5)
    assert props["ndim"] == 2
    assert props["size"] == 20
    assert props["dtype"] == "int64"
    assert props["itemsize"] == 8
    assert props["nbytes"] == 160
    assert props["strides"] == (40, 8)
    assert props["is_c_contiguous"] is True
    assert props["is_view"] is False


def test_generate_creation_code():
    code = generate_creation_code("zeros", (3, 4), dtype="float32")
    assert "np.zeros((3, 4)" in code
    assert "float32" in code

    code_rand = generate_creation_code("random", (2, 2), distribution="normal", seed=99)
    assert "default_rng" in code_rand
    assert "seed=99" in code_rand
