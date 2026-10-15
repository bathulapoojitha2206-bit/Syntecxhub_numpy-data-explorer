"""Module for NumPy array creation and metadata inspection.

Provides functions to create arrays using various NumPy constructors and generators,
as well as detailed metadata extraction and code generation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def create_from_data(
    data: Union[List, Tuple, np.ndarray],
    dtype: Optional[Union[str, np.dtype]] = None
) -> np.ndarray:
    """Create a NumPy array from nested lists, tuples, or existing sequences.

    Args:
        data: Sequence of numbers or nested sequence.
        dtype: Desired data type (e.g. 'int32', 'float64', None for auto).

    Returns:
        np.ndarray: The created NumPy array.
    """
    return np.array(data, dtype=dtype)


def create_zeros(
    shape: Union[int, Tuple[int, ...]],
    dtype: Union[str, np.dtype] = float
) -> np.ndarray:
    """Create an array filled with zeros.

    Args:
        shape: Dimension tuple, e.g. (3, 4) or integer for 1D.
        dtype: Data type.

    Returns:
        np.ndarray: Zeros array.
    """
    return np.zeros(shape, dtype=dtype)


def create_ones(
    shape: Union[int, Tuple[int, ...]],
    dtype: Union[str, np.dtype] = float
) -> np.ndarray:
    """Create an array filled with ones.

    Args:
        shape: Dimension tuple, e.g. (3, 4) or integer for 1D.
        dtype: Data type.

    Returns:
        np.ndarray: Ones array.
    """
    return np.ones(shape, dtype=dtype)


def create_empty(
    shape: Union[int, Tuple[int, ...]],
    dtype: Union[str, np.dtype] = float
) -> Tuple[np.ndarray, str]:
    """Create an uninitialized array without setting values.

    Args:
        shape: Dimension tuple.
        dtype: Data type.

    Returns:
        Tuple containing:
            np.ndarray: The uninitialized array.
            str: Warning message about uninitialized memory.
    """
    warning_msg = (
        "WARNING: np.empty() allocates raw memory without initializing values! "
        "Its initial values are uninitialized (garbage values) from arbitrary bits in RAM. "
        "You must populate it before using."
    )
    arr = np.empty(shape, dtype=dtype)
    return arr, warning_msg


def create_arange(
    start: float,
    stop: float,
    step: float = 1.0,
    dtype: Optional[Union[str, np.dtype]] = None
) -> np.ndarray:
    """Create evenly spaced values within a given half-open interval [start, stop).

    Args:
        start: Start of interval.
        stop: End of interval.
        step: Spacing between values.
        dtype: Data type.

    Returns:
        np.ndarray: 1D array of values.
    """
    if step == 0:
        raise ValueError("Step cannot be zero in arange.")
    return np.arange(start, stop, step, dtype=dtype)


def create_linspace(
    start: float,
    stop: float,
    num: int = 50,
    endpoint: bool = True,
    dtype: Optional[Union[str, np.dtype]] = None
) -> np.ndarray:
    """Create evenly spaced numbers over a specified closed or half-open interval.

    Args:
        start: Starting value.
        stop: End value.
        num: Number of samples to generate.
        endpoint: If True, stop is the last sample.
        dtype: Data type.

    Returns:
        np.ndarray: Evenly spaced 1D array.
    """
    if num <= 0:
        raise ValueError("Number of samples must be positive.")
    return np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)


def create_eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Union[str, np.dtype] = float
) -> np.ndarray:
    """Create a 2D array with ones on the diagonal and zeros elsewhere.

    Args:
        N: Number of rows.
        M: Number of columns (defaults to N if None).
        k: Index of diagonal: 0 (main diagonal), >0 (upper), <0 (lower).
        dtype: Data type.

    Returns:
        np.ndarray: 2D identity-like matrix.
    """
    if N <= 0:
        raise ValueError("N (rows) must be greater than zero.")
    return np.eye(N, M=M, k=k, dtype=dtype)


def create_full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Any,
    dtype: Optional[Union[str, np.dtype]] = None
) -> np.ndarray:
    """Create an array of given shape filled with fill_value.

    Args:
        shape: Dimension tuple.
        fill_value: Value to populate.
        dtype: Data type.

    Returns:
        np.ndarray: Filled array.
    """
    return np.full(shape, fill_value, dtype=dtype)


def create_random_array(
    shape: Union[int, Tuple[int, ...]],
    distribution: str = "uniform",
    low: float = 0.0,
    high: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    seed: Optional[int] = None,
    dtype: str = "float64"
) -> np.ndarray:
    """Generate random arrays using modern NumPy default_rng.

    Args:
        shape: Output dimensions.
        distribution: 'uniform', 'normal', or 'integers'.
        low: Lower bound for uniform/integers.
        high: Upper bound for uniform/integers.
        mean: Mean for normal distribution.
        std: Standard deviation for normal distribution.
        seed: Random seed for reproducibility.
        dtype: Target data type.

    Returns:
        np.ndarray: Random array.
    """
    rng = np.random.default_rng(seed)
    
    if distribution == "uniform":
        arr = rng.uniform(low=low, high=high, size=shape)
    elif distribution == "normal":
        arr = rng.normal(loc=mean, scale=max(std, 1e-6), size=shape)
    elif distribution == "integers":
        int_low = int(low)
        int_high = int(high)
        if int_low >= int_high:
            int_high = int_low + 1
        arr = rng.integers(low=int_low, high=int_high, size=shape)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    if dtype.startswith("int") and distribution != "integers":
        return arr.astype(dtype)
    elif dtype.startswith("float"):
        return arr.astype(dtype)
    return arr


def get_array_properties(arr: np.ndarray) -> Dict[str, Any]:
    """Inspect and extract complete memory and structural properties of an array.

    Args:
        arr: Target NumPy array.

    Returns:
        Dict with shape, ndim, size, dtype, itemsize, nbytes, strides, flags, memory_formatted.
    """
    nbytes = int(arr.nbytes)
    if nbytes < 1024:
        mem_str = f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        mem_str = f"{nbytes / 1024:.2f} KB ({nbytes:,} bytes)"
    else:
        mem_str = f"{nbytes / (1024 * 1024):.2f} MB ({nbytes:,} bytes)"

    return {
        "shape": arr.shape,
        "ndim": arr.ndim,
        "size": arr.size,
        "dtype": str(arr.dtype),
        "itemsize": arr.itemsize,
        "nbytes": nbytes,
        "strides": arr.strides,
        "is_c_contiguous": arr.flags["C_CONTIGUOUS"],
        "is_f_contiguous": arr.flags["F_CONTIGUOUS"],
        "is_writeable": arr.flags["WRITEABLE"],
        "is_view": arr.base is not None,
        "memory_formatted": mem_str,
    }


def generate_creation_code(
    method: str,
    shape: Union[int, Tuple[int, ...]],
    dtype: Optional[str] = None,
    **kwargs: Any
) -> str:
    """Generate copy-pasteable Python code for array creation.

    Args:
        method: Method name, e.g. 'zeros', 'ones', 'arange', etc.
        shape: Array shape.
        dtype: Data type string.
        **kwargs: Additional parameters.

    Returns:
        str: Python code string.
    """
    dtype_str = f", dtype='{dtype}'" if dtype else ""
    if method == "zeros":
        return f"import numpy as np\narr = np.zeros({shape}{dtype_str})"
    elif method == "ones":
        return f"import numpy as np\narr = np.ones({shape}{dtype_str})"
    elif method == "empty":
        return f"import numpy as np\narr = np.empty({shape}{dtype_str})  # Uninitialized raw RAM"
    elif method == "arange":
        start = kwargs.get("start", 0)
        stop = kwargs.get("stop", 10)
        step = kwargs.get("step", 1)
        return f"import numpy as np\narr = np.arange({start}, {stop}, {step}{dtype_str})"
    elif method == "linspace":
        start = kwargs.get("start", 0)
        stop = kwargs.get("stop", 1)
        num = kwargs.get("num", 10)
        endpoint = kwargs.get("endpoint", True)
        return f"import numpy as np\narr = np.linspace({start}, {stop}, num={num}, endpoint={endpoint}{dtype_str})"
    elif method == "eye":
        N = shape[0] if isinstance(shape, tuple) else shape
        M = shape[1] if isinstance(shape, tuple) and len(shape) > 1 else None
        k = kwargs.get("k", 0)
        m_str = f", M={M}" if M else ""
        return f"import numpy as np\narr = np.eye(N={N}{m_str}, k={k}{dtype_str})"
    elif method == "full":
        val = kwargs.get("fill_value", 7)
        return f"import numpy as np\narr = np.full({shape}, fill_value={val}{dtype_str})"
    elif method == "random":
        dist = kwargs.get("distribution", "uniform")
        seed = kwargs.get("seed", 42)
        if dist == "uniform":
            low = kwargs.get("low", 0.0)
            high = kwargs.get("high", 1.0)
            return (
                f"import numpy as np\n"
                f"rng = np.random.default_rng(seed={seed})\n"
                f"arr = rng.uniform(low={low}, high={high}, size={shape})"
            )
        elif dist == "normal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            return (
                f"import numpy as np\n"
                f"rng = np.random.default_rng(seed={seed})\n"
                f"arr = rng.normal(loc={mean}, scale={std}, size={shape})"
            )
        else:
            low = kwargs.get("low", 1)
            high = kwargs.get("high", 100)
            return (
                f"import numpy as np\n"
                f"rng = np.random.default_rng(seed={seed})\n"
                f"arr = rng.integers(low={low}, high={high}, size={shape})"
            )
    return "import numpy as np\narr = np.array(...)"
