"""Module for NumPy indexing, slicing, view/copy demonstrations, and filtering.

Provides robust indexing operations, safe slice parsing, boolean masking,
fancy indexing, diagonal extraction, array reversal, and mutation tracking
for views vs deep copies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def safe_index_1d(arr: np.ndarray, index: int) -> Tuple[Any, str]:
    """Extract element at 1D index with bounds checking.

    Args:
        arr: 1D NumPy array.
        index: Positive or negative index.

    Returns:
        Tuple of (result_element, code_string).
    """
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, received {arr.ndim}D array.")
    if index < -len(arr) or index >= len(arr):
        raise IndexError(f"Index {index} is out of bounds for axis 0 with size {len(arr)}.")
    
    val = arr[index]
    code = f"element = arr[{index}]"
    return val, code


def safe_index_2d(arr: np.ndarray, row: int, col: int) -> Tuple[Any, str]:
    """Extract element at (row, col) with bounds checking.

    Args:
        arr: 2D NumPy array.
        row: Row index.
        col: Column index.

    Returns:
        Tuple of (result_element, code_string).
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, received {arr.ndim}D array.")
    n_rows, n_cols = arr.shape
    if row < -n_rows or row >= n_rows:
        raise IndexError(f"Row index {row} is out of bounds for axis 0 with size {n_rows}.")
    if col < -n_cols or col >= n_cols:
        raise IndexError(f"Column index {col} is out of bounds for axis 1 with size {n_cols}.")

    val = arr[row, col]
    code = f"element = arr[{row}, {col}]"
    return val, code


def slice_array(
    arr: np.ndarray,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    axis: int = 0
) -> Tuple[np.ndarray, str]:
    """Perform basic or step slicing along a specified axis.

    Args:
        arr: Input array.
        start: Start index.
        stop: Stop index.
        step: Step increment (cannot be 0).
        axis: Axis to slice along.

    Returns:
        Tuple of (sliced_array, code_string).
    """
    if step == 0:
        raise ValueError("Slice step cannot be zero.")
    
    slc = slice(start, stop, step)
    slices = [slice(None)] * arr.ndim
    if 0 <= axis < arr.ndim:
        slices[axis] = slc
    else:
        raise ValueError(f"Axis {axis} is out of bounds for {arr.ndim}D array.")

    res = arr[tuple(slices)]
    
    start_str = "" if start is None else str(start)
    stop_str = "" if stop is None else str(stop)
    step_str = f":{step}" if step is not None else ""
    slice_repr = f"{start_str}:{stop_str}{step_str}"

    if arr.ndim == 1:
        code = f"sliced = arr[{slice_repr}]"
    elif arr.ndim == 2:
        if axis == 0:
            code = f"sliced = arr[{slice_repr}, :]"
        else:
            code = f"sliced = arr[:, {slice_repr}]"
    else:
        code = f"sliced = arr[{slice_repr}, ...]"

    return res, code


def slice_2d_submatrix(
    arr: np.ndarray,
    r_start: Optional[int] = None,
    r_stop: Optional[int] = None,
    r_step: Optional[int] = None,
    c_start: Optional[int] = None,
    c_stop: Optional[int] = None,
    c_step: Optional[int] = None
) -> Tuple[np.ndarray, str]:
    """Slice both rows and columns of a 2D array simultaneously.

    Args:
        arr: 2D array.
        r_start, r_stop, r_step: Row slicing parameters.
        c_start, c_stop, c_step: Column slicing parameters.

    Returns:
        Tuple of (submatrix, code_string).
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, received {arr.ndim}D array.")
    if r_step == 0 or c_step == 0:
        raise ValueError("Slice step cannot be zero.")

    r_slc = slice(r_start, r_stop, r_step)
    c_slc = slice(c_start, c_stop, c_step)
    res = arr[r_slc, c_slc]

    r_s = f"{'' if r_start is None else r_start}:{'' if r_stop is None else r_stop}{f':{r_step}' if r_step is not None else ''}"
    c_s = f"{'' if c_start is None else c_start}:{'' if c_stop is None else c_stop}{f':{c_step}' if c_step is not None else ''}"
    code = f"submatrix = arr[{r_s}, {c_s}]"
    return res, code


def select_row_or_col(
    arr: np.ndarray,
    index: int,
    axis: int = 0
) -> Tuple[np.ndarray, str]:
    """Extract an entire row or column from a 2D array.

    Args:
        arr: 2D array.
        index: Row or column index.
        axis: 0 for row selection (arr[index, :]), 1 for col selection (arr[:, index]).

    Returns:
        Tuple of (selected_vector, code_string).
    """
    if arr.ndim != 2:
        raise ValueError("Row/Column selection requires a 2D array.")
    
    n_rows, n_cols = arr.shape
    if axis == 0:
        if index < -n_rows or index >= n_rows:
            raise IndexError(f"Row {index} out of bounds for {n_rows} rows.")
        res = arr[index, :]
        code = f"row = arr[{index}, :]"
    elif axis == 1:
        if index < -n_cols or index >= n_cols:
            raise IndexError(f"Column {index} out of bounds for {n_cols} columns.")
        res = arr[:, index]
        code = f"col = arr[:, {index}]"
    else:
        raise ValueError("Axis must be 0 (row) or 1 (column).")
    
    return res, code


def filter_boolean(
    arr: np.ndarray,
    condition: str,
    threshold: float,
    upper_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Filter array elements using boolean indexing.

    Args:
        arr: Input array.
        condition: '>', '<', '>=', '<=', '==', '!=', 'between', 'even', 'odd'.
        threshold: Main numerical threshold.
        upper_threshold: Second threshold for 'between'.

    Returns:
        Tuple of (filtered_array, boolean_mask, code_string).
    """
    if condition == ">":
        mask = arr > threshold
        code = f"mask = arr > {threshold}\nresult = arr[mask]"
    elif condition == "<":
        mask = arr < threshold
        code = f"mask = arr < {threshold}\nresult = arr[mask]"
    elif condition == ">=":
        mask = arr >= threshold
        code = f"mask = arr >= {threshold}\nresult = arr[mask]"
    elif condition == "<=":
        mask = arr <= threshold
        code = f"mask = arr <= {threshold}\nresult = arr[mask]"
    elif condition == "==":
        mask = arr == threshold
        code = f"mask = arr == {threshold}\nresult = arr[mask]"
    elif condition == "!=":
        mask = arr != threshold
        code = f"mask = arr != {threshold}\nresult = arr[mask]"
    elif condition == "between":
        if upper_threshold is None:
            upper_threshold = threshold + 10
        low = min(threshold, upper_threshold)
        high = max(threshold, upper_threshold)
        mask = (arr >= low) & (arr <= high)
        code = f"mask = (arr >= {low}) & (arr <= {high})\nresult = arr[mask]"
    elif condition == "even":
        mask = (arr % 2 == 0)
        code = "mask = (arr % 2 == 0)\nresult = arr[mask]"
    elif condition == "odd":
        mask = (arr % 2 != 0)
        code = "mask = (arr % 2 != 0)\nresult = arr[mask]"
    else:
        raise ValueError(f"Unsupported condition: {condition}")

    return arr[mask], mask, code


def fancy_indexing(
    arr: np.ndarray,
    indices: List[int],
    axis: int = 0
) -> Tuple[np.ndarray, str]:
    """Extract specific non-contiguous rows, columns, or elements using fancy indexing.

    Args:
        arr: Input array.
        indices: List of integer indices.
        axis: Axis to index along.

    Returns:
        Tuple of (result_array, code_string).
    """
    idx_arr = np.array(indices, dtype=int)
    if arr.ndim == 1:
        res = arr[idx_arr]
        code = f"indices = {indices}\nresult = arr[indices]"
    elif arr.ndim == 2:
        if axis == 0:
            res = arr[idx_arr, :]
            code = f"row_indices = {indices}\nresult = arr[row_indices, :]"
        else:
            res = arr[:, idx_arr]
            code = f"col_indices = {indices}\nresult = arr[:, col_indices]"
    else:
        res = arr.take(idx_arr, axis=axis)
        code = f"result = np.take(arr, indices={indices}, axis={axis})"

    return res, code


def reverse_array(arr: np.ndarray, axis: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """Reverse an array along a specific axis or all axes.

    Args:
        arr: Input array.
        axis: Axis to reverse (None reverses all axes).

    Returns:
        Tuple of (reversed_array, code_string).
    """
    if axis is None:
        slices = tuple(slice(None, None, -1) for _ in range(arr.ndim))
        res = arr[slices]
        code = "result = arr[" + ", ".join(["::-1"] * arr.ndim) + "]"
    else:
        if axis < 0 or axis >= arr.ndim:
            raise ValueError(f"Axis {axis} out of bounds for {arr.ndim}D array.")
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(None, None, -1)
        res = arr[tuple(slices)]
        code = f"result = np.flip(arr, axis={axis})"

    return res, code


def extract_diagonal(arr: np.ndarray, offset: int = 0) -> Tuple[np.ndarray, str]:
    """Extract diagonal elements from a 2D array.

    Args:
        arr: 2D array.
        offset: 0 for main diagonal, >0 for upper, <0 for lower.

    Returns:
        Tuple of (diagonal_array, code_string).
    """
    if arr.ndim != 2:
        raise ValueError("Diagonal extraction requires a 2D array.")
    diag = np.diag(arr, k=offset)
    code = f"diag = np.diag(arr, k={offset})"
    return diag, code


def demonstrate_view_vs_copy(
    base_array: np.ndarray,
    slice_start: int = 0,
    slice_stop: int = 2,
    mutation_value: float = 999.0
) -> Dict[str, Any]:
    """Demonstrate the crucial difference between a NumPy view and a deep copy.

    Args:
        base_array: Original array to experiment with.
        slice_start: Slice start index.
        slice_stop: Slice stop index.
        mutation_value: Value to write into the sliced segment.

    Returns:
        Dict detailing before/after states, memory base checks, and educational notes.
    """
    # 1. View experiment
    orig_for_view = base_array.copy()
    view_slice = orig_for_view[slice_start:slice_stop]
    is_view_base = view_slice.base is orig_for_view
    
    # Mutate the view
    view_slice[:] = mutation_value
    view_mutated_original = orig_for_view.copy()

    # 2. Copy experiment
    orig_for_copy = base_array.copy()
    copy_slice = orig_for_copy[slice_start:slice_stop].copy()
    is_copy_base = copy_slice.base is None
    
    # Mutate the copy
    copy_slice[:] = mutation_value
    copy_mutated_original = orig_for_copy.copy()

    return {
        "original_initial": base_array.copy(),
        "slice_range": (slice_start, slice_stop),
        "mutation_value": mutation_value,
        # View results
        "view_shares_memory": is_view_base,
        "view_modified_original": view_mutated_original,
        "view_explanation": (
            "A basic slice (arr[start:stop]) creates a VIEW. It shares the underlying "
            "data buffer with the original array. Modifying the view DIRECTLY mutates "
            "the original array in place!"
        ),
        # Copy results
        "copy_shares_memory": not is_copy_base,
        "copy_modified_original": copy_mutated_original,
        "copy_explanation": (
            "Calling .copy() creates an entirely independent array in a fresh memory block. "
            "Modifying the copy has ZERO effect on the original array."
        ),
        "code_example": (
            "# View Behavior:\n"
            "view = arr[0:2]\n"
            "view[:] = 999  # WARNING: This alters 'arr'!\n\n"
            "# Copy Behavior (Safe):\n"
            "independent = arr[0:2].copy()\n"
            "independent[:] = 999  # 'arr' is completely protected!"
        )
    }
