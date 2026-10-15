"""Module for NumPy element-wise mathematical operations and safe numerical calculations.

Provides functions for arithmetic, power, trigonometric/exponential, logarithmic,
rounding, comparative, and conditional (np.where) evaluations, complete with
division-by-zero safeguards and educational metadata.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


def compute_arithmetic(
    a: np.ndarray,
    b: Union[np.ndarray, float, int],
    operation: str
) -> Tuple[np.ndarray, str, str, Optional[str]]:
    """Perform element-wise arithmetic between arrays or array and scalar.

    Args:
        a: First operand (np.ndarray).
        b: Second operand (np.ndarray or scalar).
        operation: 'add', 'subtract', 'multiply', 'divide', 'floor_divide', 'mod', 'power'.

    Returns:
        Tuple containing:
            np.ndarray: Result of the calculation.
            str: Generated Python code.
            str: Explanation of the operation.
            Optional[str]: Warning message (e.g. for division by zero or NaN).
    """
    warning = None
    is_scalar = isinstance(b, (int, float, np.integer, np.floating))

    # Catch division by zero safely
    if operation in ("divide", "floor_divide", "mod"):
        if is_scalar and b == 0:
            warning = "Division by scalar zero: IEEE 754 floating point yields inf or nan."
        elif not is_scalar and np.any(b == 0):
            warning = (
                "Division by zero detected at one or more element indices: "
                "positive/zero yields +inf, negative/zero yields -inf, and 0/0 yields nan."
            )

    # Use errstate to prevent runtime warnings from cluttering stdout while capturing warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        if operation == "add":
            res = np.add(a, b)
            op_sym = "+"
            expl = "Element-wise addition adds corresponding elements from each operand."
        elif operation == "subtract":
            res = np.subtract(a, b)
            op_sym = "-"
            expl = "Element-wise subtraction computes the difference (a - b)."
        elif operation == "multiply":
            res = np.multiply(a, b)
            op_sym = "*"
            expl = "Element-wise multiplication (Hadamard product) multiplies corresponding values."
        elif operation == "divide":
            res = np.divide(a, b)
            op_sym = "/"
            expl = "Element-wise true division computes float quotients."
        elif operation == "floor_divide":
            res = np.floor_divide(a, b)
            op_sym = "//"
            expl = "Element-wise floor division rounds quotient down to nearest integer."
        elif operation == "mod":
            res = np.mod(a, b)
            op_sym = "%"
            expl = "Element-wise modulo computes remainder of division."
        elif operation == "power":
            res = np.power(a, b)
            op_sym = "**"
            expl = "Element-wise power raises each base value in 'a' to the exponent in 'b'."
        else:
            raise ValueError(f"Unknown arithmetic operation: {operation}")

    b_repr = str(b) if is_scalar else "b"
    code = f"result = a {op_sym} {b_repr}  # or np.{operation}(a, {b_repr})"
    return res, code, expl, warning


def compute_unary_math(
    a: np.ndarray,
    operation: str,
    decimals: int = 2
) -> Tuple[np.ndarray, str, str, Optional[str]]:
    """Compute element-wise unary mathematical functions.

    Args:
        a: Input array.
        operation: 'sqrt', 'abs', 'exp', 'log', 'log10', 'log2', 'round', 'floor', 'ceil', 'sin', 'cos'.
        decimals: Decimal places for rounding.

    Returns:
        Tuple of (result_array, code_string, explanation_string, warning_message).
    """
    warning = None

    with np.errstate(divide="ignore", invalid="ignore"):
        if operation == "sqrt":
            if np.any(a < 0):
                warning = "Negative numbers encountered in np.sqrt()! NumPy outputs 'nan' (Not a Number)."
            res = np.sqrt(a)
            code = "result = np.sqrt(a)"
            expl = "Square root calculates the principal non-negative square root for every element."
        elif operation == "abs":
            res = np.abs(a)
            code = "result = np.abs(a)"
            expl = "Absolute value strips negative signs, mapping every number to |x|."
        elif operation == "exp":
            res = np.exp(a)
            code = "result = np.exp(a)"
            expl = "Exponential function computes e^x for each element (Euler's number ≈ 2.71828)."
        elif operation == "log":
            if np.any(a <= 0):
                warning = "Values <= 0 in np.log(): log(0) gives -inf; negative numbers give nan."
            res = np.log(a)
            code = "result = np.log(a)"
            expl = "Natural logarithm (base e). Inverse of np.exp."
        elif operation == "log10":
            if np.any(a <= 0):
                warning = "Values <= 0 in np.log10(): log10(0) gives -inf; negative numbers give nan."
            res = np.log10(a)
            code = "result = np.log10(a)"
            expl = "Common logarithm (base 10)."
        elif operation == "log2":
            if np.any(a <= 0):
                warning = "Values <= 0 in np.log2(): log2(0) gives -inf; negative numbers give nan."
            res = np.log2(a)
            code = "result = np.log2(a)"
            expl = "Binary logarithm (base 2), common in computer science and information theory."
        elif operation == "round":
            res = np.round(a, decimals=decimals)
            code = f"result = np.round(a, decimals={decimals})"
            expl = f"Rounds elements to {decimals} decimal places using IEEE 754 round-to-even."
        elif operation == "floor":
            res = np.floor(a)
            code = "result = np.floor(a)"
            expl = "Rounds elements down towards negative infinity."
        elif operation == "ceil":
            res = np.ceil(a)
            code = "result = np.ceil(a)"
            expl = "Rounds elements up towards positive infinity."
        elif operation == "sin":
            res = np.sin(a)
            code = "result = np.sin(a)"
            expl = "Trigonometric sine (input in radians)."
        elif operation == "cos":
            res = np.cos(a)
            code = "result = np.cos(a)"
            expl = "Trigonometric cosine (input in radians)."
        else:
            raise ValueError(f"Unknown unary math operation: {operation}")

    return res, code, expl, warning


def compute_comparison(
    a: np.ndarray,
    b: Union[np.ndarray, float, int],
    operator: str
) -> Tuple[np.ndarray, str, str]:
    """Perform element-wise comparison between arrays or array and scalar.

    Args:
        a: Array operand.
        b: Array or scalar operand.
        operator: '>', '<', '>=', '<=', '==', '!='.

    Returns:
        Tuple of (boolean_result_array, code_string, explanation).
    """
    is_scalar = isinstance(b, (int, float, np.integer, np.floating))
    b_repr = str(b) if is_scalar else "b"

    if operator == ">":
        res = a > b
    elif operator == "<":
        res = a < b
    elif operator == ">=":
        res = a >= b
    elif operator == "<=":
        res = a <= b
    elif operator == "==":
        res = a == b
    elif operator == "!=":
        res = a != b
    else:
        raise ValueError(f"Unsupported comparison operator: {operator}")

    code = f"mask = a {operator} {b_repr}"
    expl = f"Evaluates whether each element in 'a' {operator} {b_repr}, producing a boolean array."
    return res, code, expl


def compute_where(
    condition: np.ndarray,
    x: Union[np.ndarray, float, int],
    y: Union[np.ndarray, float, int]
) -> Tuple[np.ndarray, str, str]:
    """Execute conditional vectorized selection using np.where(condition, x, y).

    Args:
        condition: Boolean array mask.
        x: Values chosen when condition is True.
        y: Values chosen when condition is False.

    Returns:
        Tuple of (result_array, code_string, explanation).
    """
    res = np.where(condition, x, y)
    code = "result = np.where(condition, x, y)"
    expl = (
        "np.where(condition, x, y) is the vectorized equivalent of ternary (x if condition else y). "
        "Where condition is True, values from 'x' are selected; where False, values from 'y' are taken."
    )
    return res, code, expl


def compute_elementwise_min_max(
    a: np.ndarray,
    b: Union[np.ndarray, float, int],
    mode: str = "max"
) -> Tuple[np.ndarray, str, str]:
    """Compute element-wise minimum or maximum between two arrays or array and scalar.

    Args:
        a: First operand.
        b: Second operand.
        mode: 'max' for np.maximum, 'min' for np.minimum.

    Returns:
        Tuple of (result_array, code_string, explanation).
    """
    if mode == "max":
        res = np.maximum(a, b)
        fn = "maximum"
        expl = "np.maximum compares two arrays element-wise and returns the larger element at each position."
    else:
        res = np.minimum(a, b)
        fn = "minimum"
        expl = "np.minimum compares two arrays element-wise and returns the smaller element at each position."

    b_repr = str(b) if isinstance(b, (int, float)) else "b"
    code = f"result = np.{fn}(a, {b_repr})"
    return res, code, expl
