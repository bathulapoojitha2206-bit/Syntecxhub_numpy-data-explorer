"""Unit tests for secure NumPy file operations."""

import numpy as np
import pytest
from src.file_operations import (
    serialize_to_npy,
    serialize_to_npz,
    serialize_to_csv_txt,
    deserialize_from_bytes,
    get_storage_format_comparison,
)


def test_npy_roundtrip():
    orig = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    raw_bytes = serialize_to_npy(orig)
    assert raw_bytes.startswith(b"\x93NUMPY")

    loaded_info = deserialize_from_bytes(raw_bytes, "test_matrix.npy")
    assert loaded_info["format"] == "npy"
    assert loaded_info["is_multi"] is False
    loaded_arr = loaded_info["primary_array"]
    assert np.allclose(orig, loaded_arr)
    assert loaded_arr.dtype == orig.dtype


def test_npz_roundtrip():
    dict_arrays = {
        "features": np.arange(10, dtype=float),
        "labels": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    }
    raw_bytes = serialize_to_npz(dict_arrays, compressed=True)
    assert raw_bytes.startswith(b"PK\x03\x04")

    loaded_info = deserialize_from_bytes(raw_bytes, "dataset.npz")
    assert loaded_info["format"] == "npz"
    assert loaded_info["is_multi"] is True
    assert set(loaded_info["keys"]) == {"features", "labels"}
    assert np.array_equal(loaded_info["arrays"]["labels"], dict_arrays["labels"])


def test_csv_txt_roundtrip():
    orig = np.array([[10.0, 20.0], [30.0, 40.0]])
    csv_str = serialize_to_csv_txt(orig, delimiter=",", fmt="%.2f")
    assert "10.00,20.00" in csv_str

    raw_bytes = csv_str.encode("utf-8")
    loaded_info = deserialize_from_bytes(raw_bytes, "export.csv")
    assert loaded_info["format"] == "txt/csv"
    assert np.allclose(orig, loaded_info["primary_array"])


def test_invalid_header_rejection():
    corrupted_npy = b"FAKE_HEADER_NOT_NUMPY_DATA"
    with pytest.raises(ValueError, match="Corrupted or invalid .npy file header"):
        deserialize_from_bytes(corrupted_npy, "bad.npy")

    corrupted_npz = b"NOT_A_ZIP_ARCHIVE"
    with pytest.raises(ValueError, match="Corrupted or invalid .npz file header"):
        deserialize_from_bytes(corrupted_npz, "bad.npz")


def test_oversized_file_rejection():
    small_limit = 50  # 50 bytes limit
    payload = b"X" * 100
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        deserialize_from_bytes(payload, "large.npy", max_size_bytes=small_limit)


def test_format_comparison_table():
    comp = get_storage_format_comparison()
    assert len(comp) == 3
    formats = [item["Format"] for item in comp]
    assert any(".npy" in f for f in formats)
    assert any(".npz" in f for f in formats)
    assert any(".csv" in f for f in formats)
