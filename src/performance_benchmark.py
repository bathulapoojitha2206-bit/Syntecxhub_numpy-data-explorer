"""Module for fair, robust, and reproducible performance benchmarking: NumPy vs Python Lists.

Executes micro-benchmarks with warm-up iterations, collects multiple samples, computes
median execution times, verifies numerical equivalence, and measures memory footprint.
Includes clear architectural explanations of why SIMD, contiguous memory buffers,
and C-level loops outperform dynamic Python pointer arrays.
"""

import gc
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def measure_memory_footprint(size: int, dtype: str = "float64") -> Dict[str, Any]:
    """Measure and compare the true memory consumption of Python lists vs NumPy arrays.

    Note on Python list memory:
        A Python list stores references (pointers, 8 bytes each on 64-bit systems) to
        independent PyObject heap allocations. A Python float object typically costs 24 bytes.
        Therefore, each list element costs roughly 8 bytes (pointer) + 24 bytes (PyObject) = 32 bytes.
        A NumPy float64 array stores raw contiguous 64-bit IEEE 754 floats (exactly 8 bytes per element).

    Args:
        size: Number of elements.
        dtype: NumPy data type.

    Returns:
        Dict with list_bytes, numpy_bytes, memory_saving_ratio, and explanation.
    """
    # Sample list and array
    py_list = [float(i) for i in range(min(size, 10_000))]
    np_arr = np.array(py_list, dtype=dtype)

    # Overhead calculation scaled to full size
    py_float_size = sys.getsizeof(1.0)  # usually 24 bytes
    pointer_size = 8  # 64-bit pointer
    list_header = sys.getsizeof([])  # container overhead ~56 bytes

    estimated_list_bytes = list_header + (pointer_size + py_float_size) * size
    numpy_itemsize = np_arr.itemsize
    numpy_header = sys.getsizeof(np.array([]))  # ~112 bytes
    estimated_numpy_bytes = numpy_header + numpy_itemsize * size

    ratio = estimated_list_bytes / max(estimated_numpy_bytes, 1)

    return {
        "size": size,
        "list_bytes": estimated_list_bytes,
        "numpy_bytes": estimated_numpy_bytes,
        "list_formatted": format_bytes(estimated_list_bytes),
        "numpy_formatted": format_bytes(estimated_numpy_bytes),
        "saving_ratio": round(ratio, 2),
        "explanation": (
            f"NumPy consumes ~{ratio:.1f}x less RAM! "
            f"Standard Python lists store an array of pointers referencing separate PyObject heap allocations "
            f"(~32 bytes/float). NumPy stores unboxed contiguous binary values (8 bytes/float)."
        )
    }


def format_bytes(num_bytes: int) -> str:
    """Format raw byte count into human-friendly string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


def _time_function(func: Callable[[], Any], warmups: int = 2, repeats: int = 5) -> Tuple[float, Any]:
    """Execute a callable with warm-up runs and return the median execution time in milliseconds."""
    # Warmup runs to prime CPU caches
    for _ in range(warmups):
        res = func()

    timings = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            res = func()
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)  # convert to milliseconds
    finally:
        if gc_was_enabled:
            gc.enable()

    median_ms = float(np.median(timings))
    return median_ms, res


def benchmark_operation(
    operation: str,
    size: int,
    warmups: int = 2,
    repeats: int = 5
) -> Dict[str, Any]:
    """Run a fair, side-by-side benchmark comparing Python Lists against NumPy Arrays.

    Args:
        operation: 'creation', 'addition', 'multiplication', 'sum', 'scalar_mult'.
        size: Number of elements.
        warmups: Number of unmeasured warm-up iterations.
        repeats: Number of measured runs to compute median.

    Returns:
        Dict with list_time_ms, numpy_time_ms, speedup, verification_passed, and notes.
    """
    # Defensive clamp
    size = min(max(size, 100), 2_000_000)

    # Pre-generate identical baseline data outside the timing loop
    rng = np.random.default_rng(seed=42)
    sample_data_a = rng.uniform(low=1.0, high=100.0, size=size).tolist()
    sample_data_b = rng.uniform(low=1.0, high=100.0, size=size).tolist()

    np_a = np.array(sample_data_a, dtype=np.float64)
    np_b = np.array(sample_data_b, dtype=np.float64)

    verification_passed = True
    list_time_ms = 0.0
    numpy_time_ms = 0.0
    details = ""

    if operation == "creation":
        # Creating a list of floats via list comprehension vs np.arange/empty
        def py_task():
            return [float(i) for i in range(size)]
        
        def np_task():
            return np.arange(size, dtype=np.float64)

        list_time_ms, py_res = _time_function(py_task, warmups, repeats)
        numpy_time_ms, np_res = _time_function(np_task, warmups, repeats)
        verification_passed = (len(py_res) == len(np_res) == size)
        details = "Creating sequential elements [0.0 ... N-1]."

    elif operation == "addition":
        # Element-wise addition
        def py_task():
            return [x + y for x, y in zip(sample_data_a, sample_data_b)]

        def np_task():
            return np_a + np_b

        list_time_ms, py_res = _time_function(py_task, warmups, repeats)
        numpy_time_ms, np_res = _time_function(np_task, warmups, repeats)
        # Verify first and last elements match within float precision
        verification_passed = (
            abs(py_res[0] - np_res[0]) < 1e-4 and abs(py_res[-1] - np_res[-1]) < 1e-4
        )
        details = "Element-wise addition: a[i] + b[i] across all elements."

    elif operation == "multiplication":
        # Element-wise multiplication
        def py_task():
            return [x * y for x, y in zip(sample_data_a, sample_data_b)]

        def np_task():
            return np_a * np_b

        list_time_ms, py_res = _time_function(py_task, warmups, repeats)
        numpy_time_ms, np_res = _time_function(np_task, warmups, repeats)
        verification_passed = (
            abs(py_res[0] - np_res[0]) < 1e-3 and abs(py_res[-1] - np_res[-1]) < 1e-3
        )
        details = "Element-wise multiplication: a[i] * b[i]."

    elif operation == "sum":
        # Global aggregation / reduction
        def py_task():
            return sum(sample_data_a)

        def np_task():
            return np.sum(np_a)

        list_time_ms, py_res = _time_function(py_task, warmups, repeats)
        numpy_time_ms, np_res = _time_function(np_task, warmups, repeats)
        # Verify sum matches within relative tolerance
        rel_diff = abs(py_res - np_res) / max(abs(py_res), 1.0)
        verification_passed = (rel_diff < 1e-3)
        details = f"Summing {size:,} elements: Python sum() vs np.sum()."

    elif operation == "scalar_mult":
        # Scaling by constant
        factor = 2.5
        def py_task():
            return [x * factor for x in sample_data_a]

        def np_task():
            return np_a * factor

        list_time_ms, py_res = _time_function(py_task, warmups, repeats)
        numpy_time_ms, np_res = _time_function(np_task, warmups, repeats)
        verification_passed = (abs(py_res[0] - np_res[0]) < 1e-4)
        details = f"Scaling every element by scalar constant {factor}."

    else:
        raise ValueError(f"Unknown benchmark operation: {operation}")

    speedup = list_time_ms / max(numpy_time_ms, 0.0001)

    return {
        "operation": operation,
        "size": size,
        "list_time_ms": round(list_time_ms, 4),
        "numpy_time_ms": round(numpy_time_ms, 4),
        "speedup_ratio": round(speedup, 2),
        "faster": "NumPy" if speedup >= 1.0 else "Python List",
        "verification_passed": bool(verification_passed),
        "details": details,
        "why_it_happened": get_benchmark_explanation(operation, speedup)
    }


def get_benchmark_explanation(operation: str, speedup: float) -> str:
    """Provide educational rationale for benchmark performance discrepancy."""
    if speedup > 1.0:
        return (
            f"NumPy achieved a {speedup:.1f}x speedup because its vectorized loops run in compiled C. "
            "It avoids Python interpreter overhead (bytecode evaluation, dynamic type checking, "
            "and pointer chasing) while utilizing contiguous memory buffers and modern CPU SIMD vector registers."
        )
    else:
        return (
            "At very small sizes, Python list overhead can occasionally be competitive with NumPy "
            "due to NumPy's tiny C-wrapper invocation overhead. As array sizes grow, NumPy's vectorization dominates."
        )


def run_scaling_benchmark(
    operation: str,
    sizes: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Run benchmark across an increasing sequence of sizes for scaling curves.

    Args:
        operation: Target operation.
        sizes: List of sizes (defaults to [1_000, 10_000, 100_000, 500_000]).

    Returns:
        List of benchmark result dicts.
    """
    if sizes is None:
        sizes = [1_000, 10_000, 100_000, 500_000]

    results = []
    for s in sizes:
        res = benchmark_operation(operation, size=s, warmups=1, repeats=3)
        results.append(res)
    return results
