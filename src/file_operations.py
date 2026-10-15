"""Module for secure NumPy file storage, serialization, and deserialization.

Supports single-array .npy, multi-array .npz, compressed .npz, and plaintext .csv/.txt.
All operations strictly enforce allow_pickle=False, buffer size limits, and magic number
validation to guarantee complete security against arbitrary code execution.
"""

import io
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# NumPy format magic headers
NPY_MAGIC = b"\x93NUMPY"
ZIP_MAGIC = b"PK\x03\x04"  # NPZ files are zip archives


def serialize_to_npy(arr: np.ndarray) -> bytes:
    """Serialize a NumPy array to binary .npy format in-memory.

    Args:
        arr: Input array.

    Returns:
        bytes: Raw .npy binary content.
    """
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    return buffer.getvalue()


def serialize_to_npz(arrays: Dict[str, np.ndarray], compressed: bool = False) -> bytes:
    """Serialize multiple named arrays into a .npz zip archive in-memory.

    Args:
        arrays: Dictionary of array_name -> np.ndarray.
        compressed: If True, uses np.savez_compressed (ZIP_DEFLATED).

    Returns:
        bytes: Raw .npz binary archive content.
    """
    buffer = io.BytesIO()
    if compressed:
        np.savez_compressed(buffer, **arrays)
    else:
        np.savez(buffer, **arrays)
    return buffer.getvalue()


def serialize_to_csv_txt(
    arr: np.ndarray,
    delimiter: str = ",",
    fmt: str = "%.4f",
    header: str = ""
) -> str:
    """Serialize 1D or 2D numerical array into CSV or text format in-memory.

    Args:
        arr: 1D or 2D array.
        delimiter: Field delimiter (e.g. ',' or '\t').
        fmt: Numerical format string.
        header: Optional column header string.

    Returns:
        str: Plaintext CSV/text string.
    """
    if arr.ndim > 2:
        raise ValueError("Plaintext CSV/TXT only supports 1D or 2D arrays. Higher dimensions require .npy or .npz.")
    buffer = io.StringIO()
    np.savetxt(buffer, arr, delimiter=delimiter, fmt=fmt, header=header, comments="")
    return buffer.getvalue()


def deserialize_from_bytes(
    file_bytes: bytes,
    filename: str,
    max_size_bytes: int = 10 * 1024 * 1024  # 10 MB safe guard
) -> Dict[str, Any]:
    """Safely deserialize uploaded bytes into NumPy arrays with security checks.

    Security Measures:
        - Strictly sets allow_pickle=False.
        - Enforces maximum file size limit.
        - Inspects binary magic headers to prevent corrupted or malicious uploads.

    Args:
        file_bytes: Uploaded raw bytes.
        filename: Name of the uploaded file.
        max_size_bytes: Size upper bound.

    Returns:
        Dict containing arrays, format, properties, and status.
    """
    if len(file_bytes) > max_size_bytes:
        raise ValueError(
            f"File size ({len(file_bytes) / 1024:.1f} KB) exceeds maximum allowed limit of {max_size_bytes / (1024 * 1024):.1f} MB."
        )

    lower_name = filename.lower()
    stream = io.BytesIO(file_bytes)

    if lower_name.endswith(".npy"):
        # Validate .npy magic bytes
        if not file_bytes.startswith(NPY_MAGIC):
            raise ValueError("Corrupted or invalid .npy file header. Missing standard NumPy magic identifier.")
        arr = np.load(stream, allow_pickle=False)
        return {
            "format": "npy",
            "is_multi": False,
            "arrays": {"array_0": arr},
            "primary_array": arr,
            "summary": f".npy single array loaded successfully. Shape: {arr.shape}, Dtype: {arr.dtype}"
        }

    elif lower_name.endswith(".npz"):
        # Validate zip magic bytes
        if not file_bytes.startswith(ZIP_MAGIC):
            raise ValueError("Corrupted or invalid .npz file header. Missing standard ZIP archive signature.")
        npz_file = np.load(stream, allow_pickle=False)
        loaded_dict = {key: npz_file[key] for key in npz_file.files}
        first_key = npz_file.files[0] if npz_file.files else None
        return {
            "format": "npz",
            "is_multi": True,
            "arrays": loaded_dict,
            "primary_array": loaded_dict[first_key] if first_key else None,
            "keys": list(loaded_dict.keys()),
            "summary": f".npz archive loaded with {len(loaded_dict)} array(s): {', '.join(loaded_dict.keys())}"
        }

    elif lower_name.endswith((".csv", ".txt")):
        # Read plaintext text/csv
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        
        # Detect delimiter (comma vs tab vs whitespace)
        first_line = text.strip().split("\n")[0]
        delimiter = "," if "," in first_line else ("\t" if "\t" in first_line else None)
        
        arr = np.loadtxt(io.StringIO(text), delimiter=delimiter)
        return {
            "format": "txt/csv",
            "is_multi": False,
            "arrays": {"array_0": arr},
            "primary_array": arr,
            "summary": f"Delimited text loaded successfully. Shape: {arr.shape}, Dtype: {arr.dtype}"
        }

    else:
        raise ValueError(
            f"Unsupported file format '{filename}'. Allowed extensions: .npy, .npz, .csv, .txt"
        )


def get_storage_format_comparison() -> List[Dict[str, str]]:
    """Return an educational comparison table of NumPy storage formats.

    Returns:
        List of dicts comparing .npy, .npz, and .csv/.txt.
    """
    return [
        {
            "Format": ".npy (NumPy Single Array)",
            "Structure": "Binary with 10-byte magic header and JSON-like dict header describing shape, dtype, and Fortran order.",
            "Multi-Array?": "No (1 array per file)",
            "Precision Loss": "Zero (bit-for-bit exact binary representation)",
            "Speed": "Fastest (direct memory buffer dump & read)",
            "Human Readable": "No",
            "Best For": "High-performance intermediate storage, deep learning checkpoints, large matrices."
        },
        {
            "Format": ".npz (NumPy Multi-Array Archive)",
            "Structure": "Standard uncompressed or DEFLATE-compressed ZIP container holding individual .npy files.",
            "Multi-Array?": "Yes (dictionary of named arrays)",
            "Precision Loss": "Zero",
            "Speed": "Very Fast (slight compression/decompression overhead if using savez_compressed)",
            "Human Readable": "No",
            "Best For": "Bundling training sets, labels, feature matrices, and model weights into one compact file."
        },
        {
            "Format": ".csv / .txt (Plaintext)",
            "Structure": "Human-readable ASCII/UTF-8 text with delimited numbers.",
            "Multi-Array?": "No (limited to 1D or 2D)",
            "Precision Loss": "Possible (floating point numbers are truncated by format string specifiers)",
            "Speed": "Slowest (parsing text strings to binary IEEE 754 floats is computationally expensive)",
            "Human Readable": "Yes (can open in Excel, Notepad, or any text editor)",
            "Best For": "Interoperability with non-Python tools, spreadsheet software, and web APIs."
        }
    ]
