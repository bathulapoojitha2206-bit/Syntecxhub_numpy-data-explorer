"""Module for NumPy broadcasting analysis, rule validation, and visual explanation.

Implements strict right-to-left dimension alignment matching NumPy's internal
broadcasting specification, generates detailed compatibility diagnostics, and
provides real-world vectorization examples (e.g., column normalization).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def analyze_broadcasting(
    shape_a: Tuple[int, ...],
    shape_b: Tuple[int, ...]
) -> Dict[str, Any]:
    """Examine two array shapes according to NumPy's 4 fundamental broadcasting rules.

    Rules:
        1. Align shapes from right to left (trailing dimensions first).
        2. Prepend 1s to the shorter shape until both tuples have equal rank.
        3. For each aligned dimension pair (d_a, d_b):
           - If d_a == d_b, compatible (output dimension is d_a).
           - If d_a == 1, compatible (stretched along axis to match d_b).
           - If d_b == 1, compatible (stretched along axis to match d_a).
           - Otherwise, INCOMPATIBLE.
        4. If all pairs are compatible, the resulting shape is formed by max(d_a, d_b).

    Args:
        shape_a: Shape tuple of operand A.
        shape_b: Shape tuple of operand B.

    Returns:
        Dict containing compatibility status, aligned tuples, step-by-step trace,
        resulting shape, and error diagnosis if incompatible.
    """
    rank_a = len(shape_a)
    rank_b = len(shape_b)
    max_rank = max(rank_a, rank_b)

    # Pad shapes on the left with 1s to make ranks equal
    padded_a = (1,) * (max_rank - rank_a) + tuple(shape_a)
    padded_b = (1,) * (max_rank - rank_b) + tuple(shape_b)

    steps = []
    result_shape = []
    is_compatible = True
    incompatible_axis = None
    failure_detail = ""

    # Compare from left to right on the padded shapes (equivalent to right-to-left on originals)
    for idx, (da, db) in enumerate(zip(padded_a, padded_b)):
        axis_from_right = max_rank - idx
        if da == db:
            status = "Match"
            res_dim = da
            desc = f"Both dimensions equal {da} -> Output dimension is {res_dim}."
        elif da == 1:
            status = "Stretch A"
            res_dim = db
            desc = f"Operand A has size 1 -> Stretched {db} times to match B."
        elif db == 1:
            status = "Stretch B"
            res_dim = da
            desc = f"Operand B has size 1 -> Stretched {da} times to match A."
        else:
            status = "MISMATCH"
            is_compatible = False
            incompatible_axis = idx
            failure_detail = (
                f"Dimension mismatch at axis {idx} (trailing axis -{axis_from_right}): "
                f"Operand A has size {da} while Operand B has size {db}. "
                f"Neither is 1 and they are not equal."
            )
            res_dim = None
            desc = f"CONFLICT: {da} != {db} and neither is 1!"

        result_shape.append(res_dim)
        steps.append({
            "axis_index": idx,
            "axis_from_right": f"-{axis_from_right}",
            "dim_a": da,
            "dim_b": db,
            "status": status,
            "result_dim": res_dim,
            "explanation": desc
        })

    final_shape = tuple(result_shape) if is_compatible else None

    return {
        "is_compatible": is_compatible,
        "shape_a": shape_a,
        "shape_b": shape_b,
        "padded_a": padded_a,
        "padded_b": padded_b,
        "result_shape": final_shape,
        "steps": steps,
        "failure_reason": failure_detail if not is_compatible else None,
        "rules_applied": [
            "1. Shapes aligned from right-to-left (trailing dimensions match first).",
            f"2. Padded shorter shape with leading 1s: A={padded_a}, B={padded_b}.",
            "3. Evaluated whether dimensions are identical or one equals 1.",
            "4. Determined final broadcasted shape or caught incompatibility."
        ]
    }


def broadcast_and_operate(
    a: np.ndarray,
    b: np.ndarray,
    operation: str = "add"
) -> Tuple[np.ndarray, Tuple[int, ...], str]:
    """Execute broadcasted operation between two arrays with safety limits.

    Args:
        a: First array.
        b: Second array.
        operation: 'add', 'subtract', 'multiply', 'divide'.

    Returns:
        Tuple of (result_array, result_shape, code_string).
    """
    analysis = analyze_broadcasting(a.shape, b.shape)
    if not analysis["is_compatible"]:
        raise ValueError(
            f"Operands could not be broadcast together with shapes {a.shape} and {b.shape}. "
            f"{analysis['failure_reason']}"
        )

    res_shape = analysis["result_shape"]
    total_elements = np.prod(res_shape)
    if total_elements > 500_000:
        raise ValueError(
            f"Broadcast result shape {res_shape} would allocate {total_elements:,} elements. "
            f"To protect system memory, operations are capped at 500,000 elements."
        )

    if operation == "add":
        res = a + b
        sym = "+"
    elif operation == "subtract":
        res = a - b
        sym = "-"
    elif operation == "multiply":
        res = a * b
        sym = "*"
    elif operation == "divide":
        with np.errstate(divide="ignore", invalid="ignore"):
            res = a / b
        sym = "/"
    else:
        raise ValueError(f"Unknown operation: {operation}")

    code = f"# Array A shape: {a.shape}\n# Array B shape: {b.shape}\nresult = a {sym} b  # Output shape: {res.shape}"
    return res, res.shape, code


def create_broadcasting_scenarios() -> Dict[str, Dict[str, Any]]:
    """Provide structured educational broadcasting scenarios.

    Returns:
        Dict of scenario definitions with arrays, names, descriptions, and use cases.
    """
    return {
        "1. Scalar & 2D Matrix": {
            "a": np.array([[10, 20, 30], [40, 50, 60]]),
            "b": np.array(5),
            "description": "A scalar (0D) is broadcast across all rows and columns of a 2D matrix (2, 3).",
            "concept": "Scalar is treated as (1, 1) and stretched to (2, 3)."
        },
        "2. 2D Matrix & 1D Row Vector": {
            "a": np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),  # (3, 4)
            "b": np.array([10, 20, 30, 40]),  # (4,)
            "description": "Row vector (4,) is padded to (1, 4) and stretched vertically across 3 rows.",
            "concept": "Adding column offsets or feature adjustments to rows of samples."
        },
        "3. Column Vector & Row Vector (Outer Expansion)": {
            "a": np.array([[1], [2], [3]]),      # (3, 1)
            "b": np.array([[10, 20, 30, 40]]),   # (1, 4)
            "description": "Column vector (3, 1) stretched horizontally and Row vector (1, 4) stretched vertically to produce a (3, 4) grid.",
            "concept": "Efficiently computing pairwise distances, Cartesian coordinate grids, or multiplication tables without loops."
        },
        "4. Practical: Column-wise Mean Centering (Normalization)": {
            "a": np.array([
                [85, 90, 78],
                [92, 88, 84],
                [75, 82, 80],
                [88, 95, 91]
            ]),  # (4, 3) matrix of scores
            "b": np.array([[85.0, 88.75, 83.25]]),  # (1, 3) column means
            "description": "Subtract column means (1, 3) from sample data (4, 3) to center data at zero.",
            "concept": "Standard preprocessing step in Data Science, PCA, and Machine Learning."
        },
        "5. Incompatible Dimension Conflict (Educational Failure)": {
            "a": np.array([[1, 2, 3], [4, 5, 6]]),  # (2, 3)
            "b": np.array([10, 20]),                # (2,) -> padded to (1, 2)
            "description": "Shapes (2, 3) and (2,) fail because trailing dimensions 3 and 2 mismatch and neither is 1.",
            "concept": "Demonstrates why broadcasting fails and how transposing or reshaping to (2, 1) resolves it."
        }
    }
