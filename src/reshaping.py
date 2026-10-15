"""Module for NumPy array reshaping, joining, splitting, expanding, and manipulation.

Provides operations for reshape, flatten vs ravel, transpose/.T, concatenate,
stack/vstack/hstack, split/array_split, expand_dims, squeeze, repeat, tile,
sort, unique, and view-versus-copy inspection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def is_view_of(target: np.ndarray, source: np.ndarray) -> bool:
    """Determine whether target array shares underlying memory buffer with source.

    Args:
        target: Evaluated array.
        source: Potential parent array.

    Returns:
        bool: True if target is a view of source, False if it is an independent copy.
    """
    if target.base is None:
        return False
    if target.base is source:
        return True
    if source.base is not None and target.base is source.base:
        return True
    return False


def validate_reshape(current_size: int, target_shape: Tuple[int, ...]) -> Tuple[bool, str]:
    """Validate whether an array with current_size elements can be reshaped to target_shape.

    Args:
        current_size: Total element count (arr.size).
        target_shape: Proposed dimension tuple (may include a single -1).

    Returns:
        Tuple of (is_valid, reason_or_inferred_shape_message).
    """
    if any(d == 0 for d in target_shape):
        if current_size == 0:
            return True, "Valid 0-size reshape."
        return False, "Target dimension cannot be zero for non-empty array."

    neg_count = target_shape.count(-1)
    if neg_count > 1:
        return False, "Can only specify one unknown dimension (-1) in target shape."

    known_product = 1
    for d in target_shape:
        if d != -1:
            if d < 1:
                return False, f"Invalid dimension size {d}. Dimensions must be positive or -1."
            known_product *= d

    if neg_count == 1:
        if current_size % known_product != 0:
            return (
                False,
                f"Cannot infer -1 dimension: Total size {current_size} is not divisible by {known_product}."
            )
        inferred = current_size // known_product
        full_shape = tuple(inferred if d == -1 else d for d in target_shape)
        return True, f"Valid! -1 dimension inferred as {inferred}. Full shape: {full_shape}"

    if known_product != current_size:
        return (
            False,
            f"Shape mismatch: Total size is {current_size}, but target shape product is {known_product}."
        )

    return True, f"Valid! Dimensions product ({known_product}) matches size ({current_size})."


def execute_reshape(arr: np.ndarray, target_shape: Tuple[int, ...]) -> Tuple[np.ndarray, bool, str]:
    """Reshape array into target shape with validation and view tracking.

    Args:
        arr: Input array.
        target_shape: Desired shape tuple.

    Returns:
        Tuple of (reshaped_array, is_view, code_string).
    """
    valid, msg = validate_reshape(arr.size, target_shape)
    if not valid:
        raise ValueError(f"Invalid reshape request: {msg}")

    reshaped = arr.reshape(target_shape)
    view_status = is_view_of(reshaped, arr)
    code = f"reshaped = arr.reshape({target_shape})"
    return reshaped, view_status, code


def execute_flatten_vs_ravel(arr: np.ndarray, method: str = "ravel") -> Tuple[np.ndarray, bool, str, str]:
    """Compare .ravel() (view when possible) and .flatten() (always a copy).

    Args:
        arr: Input array.
        method: 'ravel' or 'flatten'.

    Returns:
        Tuple of (result_1d_array, is_view, code_string, educational_note).
    """
    if method == "ravel":
        res = arr.ravel()
        view_status = is_view_of(res, arr)
        code = "flat = arr.ravel()"
        note = (
            "arr.ravel() returns a contiguous 1D VIEW of the original array whenever possible "
            "without copying memory buffers. Changes to the raveled array mutate the original array!"
        )
    elif method == "flatten":
        res = arr.flatten()
        view_status = False  # flatten() always creates an independent deep copy
        code = "flat = arr.flatten()"
        note = (
            "arr.flatten() ALWAYS allocates fresh memory and returns a deep COPY. "
            "Mutating a flattened array never modifies the original."
        )
    else:
        raise ValueError(f"Unknown flattening method: {method}")

    return res, view_status, code, note


def execute_transpose(arr: np.ndarray, axes: Optional[Tuple[int, ...]] = None) -> Tuple[np.ndarray, bool, str]:
    """Transpose dimensions of an array.

    Args:
        arr: Input array.
        axes: Permutation of axes (None reverses axes).

    Returns:
        Tuple of (transposed_array, is_view, code_string).
    """
    if axes is not None:
        res = np.transpose(arr, axes=axes)
        code = f"transposed = np.transpose(arr, axes={axes})"
    else:
        res = arr.T
        code = "transposed = arr.T  # or np.transpose(arr)"

    view_status = is_view_of(res, arr)
    return res, view_status, code


def execute_join_operation(
    arr1: np.ndarray,
    arr2: np.ndarray,
    operation: str = "concatenate",
    axis: int = 0
) -> Tuple[np.ndarray, str]:
    """Join multiple arrays using concatenate, vstack, hstack, or stack.

    Args:
        arr1: First array.
        arr2: Second array.
        operation: 'concatenate', 'vstack', 'hstack', 'stack'.
        axis: Axis along which to join.

    Returns:
        Tuple of (joined_array, code_string).
    """
    if operation == "concatenate":
        res = np.concatenate([arr1, arr2], axis=axis)
        code = f"result = np.concatenate([arr1, arr2], axis={axis})"
    elif operation == "vstack":
        res = np.vstack([arr1, arr2])
        code = "result = np.vstack([arr1, arr2])  # vertical stacking (along rows)"
    elif operation == "hstack":
        res = np.hstack([arr1, arr2])
        code = "result = np.hstack([arr1, arr2])  # horizontal stacking (along columns)"
    elif operation == "stack":
        res = np.stack([arr1, arr2], axis=axis)
        code = f"result = np.stack([arr1, arr2], axis={axis})  # stacks along new dimension"
    else:
        raise ValueError(f"Unknown join operation: {operation}")

    return res, code


def execute_split_operation(
    arr: np.ndarray,
    sections: int,
    axis: int = 0,
    allow_unequal: bool = False
) -> Tuple[List[np.ndarray], str]:
    """Split an array into multiple sub-arrays along an axis.

    Args:
        arr: Input array.
        sections: Number of sections.
        axis: Axis to split along.
        allow_unequal: If True, uses np.array_split (tolerates non-exact divides).

    Returns:
        Tuple of (list_of_splits, code_string).
    """
    if sections <= 0:
        raise ValueError("Number of sections must be positive.")

    axis_len = arr.shape[axis]
    if not allow_unequal and axis_len % sections != 0:
        raise ValueError(
            f"Array split does not result in an equal division: "
            f"Axis {axis} has length {axis_len}, which cannot be divided evenly into {sections} sections. "
            f"Tip: Switch to 'array_split' to allow unequal splits."
        )

    if allow_unequal:
        splits = np.array_split(arr, sections, axis=axis)
        code = f"splits = np.array_split(arr, {sections}, axis={axis})"
    else:
        splits = np.split(arr, sections, axis=axis)
        code = f"splits = np.split(arr, {sections}, axis={axis})"

    return splits, code


def execute_expand_or_squeeze(
    arr: np.ndarray,
    operation: str,
    axis: int = 0
) -> Tuple[np.ndarray, bool, str]:
    """Expand dimensions (np.expand_dims) or squeeze 1-length dimensions (np.squeeze).

    Args:
        arr: Input array.
        operation: 'expand_dims' or 'squeeze'.
        axis: Target axis.

    Returns:
        Tuple of (result_array, is_view, code_string).
    """
    if operation == "expand_dims":
        res = np.expand_dims(arr, axis=axis)
        code = f"expanded = np.expand_dims(arr, axis={axis})"
    elif operation == "squeeze":
        res = np.squeeze(arr, axis=axis if axis < arr.ndim else None)
        code = f"squeezed = np.squeeze(arr)" if axis is None else f"squeezed = np.squeeze(arr, axis={axis})"
    else:
        raise ValueError(f"Unknown operation: {operation}")

    view_status = is_view_of(res, arr)
    return res, view_status, code


def execute_repeat_or_tile(
    arr: np.ndarray,
    operation: str,
    repeats: int,
    axis: Optional[int] = None
) -> Tuple[np.ndarray, str]:
    """Construct arrays via element repetition (repeat) or block tiling (tile).

    Args:
        arr: Input array.
        operation: 'repeat' or 'tile'.
        repeats: Multiplier integer.
        axis: Axis for repeat (ignored for tile).

    Returns:
        Tuple of (result_array, code_string).
    """
    if operation == "repeat":
        res = np.repeat(arr, repeats=repeats, axis=axis)
        axis_str = f", axis={axis}" if axis is not None else ""
        code = f"repeated = np.repeat(arr, repeats={repeats}{axis_str})"
    elif operation == "tile":
        res = np.tile(arr, reps=repeats)
        code = f"tiled = np.tile(arr, reps={repeats})"
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return res, code


def execute_sort_or_unique(
    arr: np.ndarray,
    operation: str,
    axis: Optional[int] = -1
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Perform sorting or distinct element extraction.

    Args:
        arr: Input array.
        operation: 'sort' or 'unique'.
        axis: Axis to sort along (used for sort).

    Returns:
        Tuple of (result_array, optional_counts, code_string).
    """
    if operation == "sort":
        res = np.sort(arr, axis=axis)
        axis_str = f", axis={axis}" if axis is not None else ""
        code = f"sorted_arr = np.sort(arr{axis_str})"
        return res, None, code
    elif operation == "unique":
        res, counts = np.unique(arr, return_counts=True)
        code = "unique_vals, counts = np.unique(arr, return_counts=True)"
        return res, counts, code
    else:
        raise ValueError(f"Unknown operation: {operation}")
