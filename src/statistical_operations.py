"""Module for NumPy statistical and axis-wise operations.

Provides comprehensive descriptive statistics, axis semantics (axis=0 down rows,
axis=1 across columns, axis=None flattened), degree-of-freedom (ddof) controls,
and analytical synthetic dataset generators.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_all_statistics(
    arr: np.ndarray,
    axis: Optional[int] = None,
    ddof: int = 0,
    percentile_q: float = 75.0
) -> Dict[str, Any]:
    """Calculate an exhaustive suite of NumPy statistical metrics.

    Args:
        arr: Input array.
        axis: Axis over which to compute (0, 1, or None for entire array).
        ddof: Delta Degrees of Freedom (0 for population, 1 for sample).
        percentile_q: Percentile rank (0 - 100).

    Returns:
        Dict mapping metric names to their calculated values and documentation.
    """
    # Defensive copy and floating conversion for safe statistical operations
    calc_arr = arr.astype(float) if np.issubdtype(arr.dtype, np.integer) else arr

    results = {}
    
    # Aggregations
    results["sum"] = np.sum(calc_arr, axis=axis)
    results["mean"] = np.mean(calc_arr, axis=axis)
    results["median"] = np.median(calc_arr, axis=axis)
    results["min"] = np.min(calc_arr, axis=axis)
    results["max"] = np.max(calc_arr, axis=axis)
    results["ptp"] = np.ptp(calc_arr, axis=axis)  # peak-to-peak (range)
    
    # Dispersion
    results["var"] = np.var(calc_arr, axis=axis, ddof=ddof)
    results["std"] = np.std(calc_arr, axis=axis, ddof=ddof)
    
    # Percentiles / Quantiles
    results["percentile"] = np.percentile(calc_arr, q=percentile_q, axis=axis)
    results["quantile_50"] = np.quantile(calc_arr, q=0.5, axis=axis)
    results["quantile_25"] = np.quantile(calc_arr, q=0.25, axis=axis)
    results["quantile_75"] = np.quantile(calc_arr, q=0.75, axis=axis)

    # Arg extrema (indices)
    try:
        results["argmin"] = np.argmin(arr, axis=axis)
        results["argmax"] = np.argmax(arr, axis=axis)
    except Exception:
        results["argmin"] = None
        results["argmax"] = None

    return results


def compute_cumulative_stats(
    arr: np.ndarray,
    axis: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative sum (np.cumsum) and cumulative product (np.cumprod).

    Args:
        arr: Input array.
        axis: Axis along which to accumulate.

    Returns:
        Tuple of (cumsum_array, cumprod_array).
    """
    c_sum = np.cumsum(arr, axis=axis)
    with np.errstate(over="ignore", invalid="ignore"):
        c_prod = np.cumprod(arr, axis=axis)
    return c_sum, c_prod


def get_axis_explanation(axis: Optional[int]) -> Dict[str, str]:
    """Provide educational explanation of NumPy axis mechanics.

    Args:
        axis: 0, 1, or None.

    Returns:
        Dict with title, description, and visual metaphor.
    """
    if axis is None:
        return {
            "title": "axis=None (Global Flattened Calculation)",
            "description": (
                "Flattens the entire array into a single 1D sequence before calculation. "
                "The result is a single scalar summary for the entire dataset."
            ),
            "code_sample": "np.mean(arr, axis=None)  # Output: scalar float",
            "arrow": "Collapses all rows and columns into one value."
        }
    elif axis == 0:
        return {
            "title": "axis=0 (Down the Rows -> Column-wise Result)",
            "description": (
                "Operations collapse along rows (vertical direction). For a 2D matrix of shape (M, N), "
                "axis=0 collapses the M rows, producing a 1D vector of length N (one value per column)."
            ),
            "code_sample": "np.mean(arr, axis=0)  # Output shape: (N,) - average per column",
            "arrow": "Vertical arrow: ↓ (aggregates each column independently)"
        }
    elif axis == 1:
        return {
            "title": "axis=1 (Across the Columns -> Row-wise Result)",
            "description": (
                "Operations collapse along columns (horizontal direction). For a 2D matrix of shape (M, N), "
                "axis=1 collapses the N columns, producing a 1D vector of length M (one value per row)."
            ),
            "code_sample": "np.mean(arr, axis=1)  # Output shape: (M,) - average per row",
            "arrow": "Horizontal arrow: → (aggregates each row independently)"
        }
    else:
        return {
            "title": f"axis={axis}",
            "description": f"Collapses along dimension index {axis}.",
            "code_sample": f"np.mean(arr, axis={axis})",
            "arrow": f"Dimension {axis}"
        }


def get_ddof_explanation(ddof: int) -> str:
    """Explain degrees of freedom for variance and standard deviation.

    Args:
        ddof: 0 or 1.

    Returns:
        Explanation string.
    """
    if ddof == 0:
        return (
            "ddof=0 (Population Standard Deviation & Variance):\n"
            "Formula: Var = Σ(x - μ)² / N\n"
            "This assumes you have access to the ENTIRE population of data. "
            "NumPy uses ddof=0 by default."
        )
    else:
        return (
            "ddof=1 (Sample Standard Deviation & Variance):\n"
            "Formula: Var = Σ(x - x̄)² / (N - 1)\n"
            "Applies Bessel's correction to remove sample variance bias when estimating "
            "an unknown population variance from a subset sample. Pandas uses ddof=1 by default."
        )


def generate_sample_datasets() -> Dict[str, Tuple[pd.DataFrame, np.ndarray, str]]:
    """Generate realistic synthetic analytical datasets for educational exploration.

    Returns:
        Dict of dataset names mapping to (DataFrame, NumPy array, description).
    """
    rng = np.random.default_rng(seed=42)

    # 1. Student Marks Dataset: 15 students, 5 subjects
    students = [f"Student_{i+1:02d}" for i in range(15)]
    subjects = ["Mathematics", "Physics", "Chemistry", "Computer Science", "English"]
    # Base marks with realistic distributions
    marks = rng.integers(low=45, high=100, size=(15, 5))
    df_students = pd.DataFrame(marks, index=students, columns=subjects)
    desc_students = (
        "Synthetic exam marks for 15 students across 5 subjects. "
        "Useful for exploring column-wise statistics (subject difficulty) "
        "versus row-wise statistics (individual student performance)."
    )

    # 2. Monthly Sales Dataset: 12 months, 4 regions
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    regions = ["North", "South", "East", "West"]
    sales = rng.integers(low=12000, high=85000, size=(12, 4))
    df_sales = pd.DataFrame(sales, index=months, columns=regions)
    desc_sales = (
        "Monthly regional sales revenue ($) across 12 months in 4 territories. "
        "Great for cumulative sum tracking (YTD revenue) and regional variance."
    )

    # 3. Product Ratings: 10 products, 4 review categories
    products = [f"Product_{chr(65+i)}" for i in range(10)]
    criteria = ["Durability", "Ergonomics", "Performance", "Value for Money"]
    ratings = np.round(rng.uniform(low=2.5, high=5.0, size=(10, 4)), 1)
    df_ratings = pd.DataFrame(ratings, index=products, columns=criteria)
    desc_ratings = (
        "Quality audit ratings (scale 1.0 to 5.0) for 10 consumer hardware products. "
        "Ideal for quantile analysis, top-performing category identification, and normalization."
    )

    return {
        "Student Marks (15 x 5)": (df_students, marks, desc_students),
        "Monthly Regional Sales (12 x 4)": (df_sales, sales, desc_sales),
        "Product Quality Ratings (10 x 4)": (df_ratings, ratings, desc_ratings),
    }
