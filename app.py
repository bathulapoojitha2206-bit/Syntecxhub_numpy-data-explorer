"""NumPy Data Explorer — Interactive Numerical Computing Lab.

A comprehensive, production-grade Streamlit web application for learning,
visualizing, experimenting with, and mastering NumPy numerical computing concepts.
"""

import io
import sys
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import core numerical computing modules
from src.array_creation import (
    create_arange,
    create_empty,
    create_eye,
    create_from_data,
    create_full,
    create_linspace,
    create_ones,
    create_random_array,
    create_zeros,
    generate_creation_code,
    get_array_properties,
)
from src.indexing_slicing import (
    demonstrate_view_vs_copy,
    extract_diagonal,
    fancy_indexing,
    filter_boolean,
    reverse_array,
    safe_index_1d,
    safe_index_2d,
    select_row_or_col,
    slice_2d_submatrix,
    slice_array,
)
from src.mathematical_operations import (
    compute_arithmetic,
    compute_comparison,
    compute_elementwise_min_max,
    compute_unary_math,
    compute_where,
)
from src.statistical_operations import (
    compute_all_statistics,
    compute_cumulative_stats,
    generate_sample_datasets,
    get_axis_explanation,
    get_ddof_explanation,
)
from src.reshaping import (
    execute_expand_or_squeeze,
    execute_flatten_vs_ravel,
    execute_join_operation,
    execute_repeat_or_tile,
    execute_reshape,
    execute_sort_or_unique,
    execute_split_operation,
    execute_transpose,
    is_view_of,
    validate_reshape,
)
from src.broadcasting import (
    analyze_broadcasting,
    broadcast_and_operate,
    create_broadcasting_scenarios,
)
from src.file_operations import (
    deserialize_from_bytes,
    get_storage_format_comparison,
    serialize_to_csv_txt,
    serialize_to_npy,
    serialize_to_npz,
)
from src.performance_benchmark import (
    benchmark_operation,
    format_bytes,
    measure_memory_footprint,
    run_scaling_benchmark,
)
from src.visualizations import (
    plot_1d_array,
    plot_2d_heatmap,
    plot_benchmark_comparison,
    plot_distribution_analysis,
    plot_memory_comparison,
    plot_scaling_curves,
    plot_sliced_submatrix,
)

# ==========================================
# Application Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="NumPy Data Explorer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern numerical lab aesthetic
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'Fira Code', monospace !important;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .metric-card .val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E293B;
    }
    
    .metric-card .lbl {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stAlert {
        border-radius: 8px;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-view {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .badge-copy {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_properties_bar(props: Dict[str, Any]):
    """Render a responsive horizontal badge-bar showing core array metadata."""
    cols = st.columns(6)
    with cols[0]:
        st.markdown(f"<div class='metric-card'><div class='val'>{props['shape']}</div><div class='lbl'>Shape</div></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"<div class='metric-card'><div class='val'>{props['ndim']}D</div><div class='lbl'>Dimensions</div></div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"<div class='metric-card'><div class='val'>{props['size']:,}</div><div class='lbl'>Elements</div></div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f"<div class='metric-card'><div class='val'>{props['dtype']}</div><div class='lbl'>Dtype</div></div>", unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f"<div class='metric-card'><div class='val'>{props['memory_formatted']}</div><div class='lbl'>RAM Footprint</div></div>", unsafe_allow_html=True)
    with cols[5]:
        view_tag = "VIEW" if props["is_view"] else "BASE"
        st.markdown(f"<div class='metric-card'><div class='val'>{view_tag}</div><div class='lbl'>Memory Status</div></div>", unsafe_allow_html=True)
    st.write("")


# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.markdown("### ⚡ NumPy Data Explorer")
st.sidebar.caption("Interactive Numerical Computing Lab v1.0")

pages = [
    "🏠 Overview & Guide",
    "🧪 1. Array Creation Lab",
    "🔍 2. Indexing & Slicing Lab",
    "➕ 3. Mathematical Operations",
    "📊 4. Statistical & Axis Lab",
    "🔄 5. Reshaping & Manipulation",
    "📡 6. Broadcasting Visualizer",
    "💾 7. NumPy File Operations",
    "⚡ 8. Performance Benchmark",
    "📈 9. Practical Data Analysis",
]

selected_page = st.sidebar.radio("Navigate Lab Modules", pages, index=0)

st.sidebar.divider()
st.sidebar.info(
    "💡 **NumPy Tip:** Vectorized operations run directly in compiled C, avoiding slow Python bytecode evaluation."
)


# ==========================================
# PAGE 0: OVERVIEW & GUIDE
# ==========================================
if selected_page == "🏠 Overview & Guide":
    st.markdown("<div class='main-title'>⚡ NumPy Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Interactive Numerical Computing Lab for Python Developers & Data Analysts</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(
            """
            Welcome to **NumPy Data Explorer**! This platform provides hands-on, visual, and mathematical
            intuition for modern numerical computing with NumPy.
            
            ### 🎯 Core Learning Objectives:
            1. **Memory & Contiguity:** Learn how `ndarray` stores homogeneous C-contiguous data buffers in RAM.
            2. **Subsetting & Views:** Grasp the subtle yet critical difference between memory views and deep copies.
            3. **Vectorized Math:** Perform element-wise mathematical operations at compiled hardware speed.
            4. **Axis Dimensionality:** Master `axis=0` (down the rows) vs `axis=1` (across the columns) aggregation.
            5. **Broadcasting Mechanics:** Demystify the 4 rules of automatic dimension expansion.
            6. **Hardware Efficiency:** Benchmark NumPy against standard Python lists to understand cache locality and SIMD.
            """
        )

        st.markdown("### 🗺️ Module Directory")
        st.markdown(
            """
            - **🧪 Array Creation:** Construct 1D, 2D, and 3D arrays via `zeros`, `ones`, `arange`, `linspace`, `eye`, and PRNGs.
            - **🔍 Indexing & Slicing:** Explore positive/negative bounds, step slicing, boolean masks, and view mutation.
            - **➕ Mathematical Operations:** Arithmetic, powers, square roots, logarithms, comparisons, and `np.where`.
            - **📊 Statistical Analysis:** Aggregations across axes, population vs sample variance (`ddof`), percentiles.
            - **🔄 Reshaping & Transposition:** `reshape`, `flatten` vs `ravel`, `vstack`, `hstack`, and `array_split`.
            - **📡 Broadcasting Engine:** Right-to-left alignment tracing with interactive step-by-step diagnostics.
            - **💾 File Serialization:** Secure `.npy`, `.npz`, and `.csv` saving and loading with `allow_pickle=False`.
            - **⚡ Performance Benchmarks:** Empirical speedup ratios and RAM memory footprint comparisons.
            - **📈 Practical Data Analysis:** Real-world case study with student marks, sales, and quality metrics.
            """
        )

    with col2:
        st.markdown("### 🧠 The NumPy Mental Model")
        st.markdown(
            """
            A standard Python list `[1.0, 2.0, 3.0]` is an **array of pointers** referencing distinct
            `PyObject` allocations scattered across the heap.
            
            In contrast, a NumPy array `np.array([1.0, 2.0, 3.0])` is a **single contiguous block of raw bytes**
            in memory paired with a lightweight Python header describing its **shape**, **dtype**, and **strides**.
            """
        )
        
        # Interactive Cheatsheet Quick Tester
        st.markdown("#### ⚡ Quick NumPy Inspector")
        sample_choice = st.selectbox(
            "Select sample array to inspect",
            ["1D Vector (10 elements)", "2D Matrix (4x5)", "3D Tensor (2x3x4)"]
        )
        if sample_choice == "1D Vector (10 elements)":
            demo_arr = np.linspace(0, 100, 10)
        elif sample_choice == "2D Matrix (4x5)":
            demo_arr = np.arange(20).reshape(4, 5)
        else:
            demo_arr = np.ones((2, 3, 4), dtype=np.float32)

        st.json(get_array_properties(demo_arr))


# ==========================================
# PAGE 1: ARRAY CREATION LAB
# ==========================================
elif selected_page == "🧪 1. Array Creation Lab":
    st.markdown("<div class='main-title'>🧪 Array Creation Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Explore NumPy constructors, memory allocation, and metadata inspection</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Lab", "Learning Guide & Common Pitfalls"])

    with tabs[0]:
        col_ctrl, col_view = st.columns([1, 2])

        with col_ctrl:
            st.markdown("#### ⚙️ Creation Controls")
            method = st.selectbox(
                "Creation Method",
                ["np.zeros", "np.ones", "np.empty", "np.arange", "np.linspace", "np.eye", "np.full", "np.random.default_rng", "Custom List/Array"]
            )

            ndim = st.radio("Dimensionality", ["1D", "2D", "3D"], horizontal=True)

            if ndim == "1D":
                dim1 = st.slider("Length (Size)", 1, 100, 8)
                shape = (dim1,)
            elif ndim == "2D":
                r = st.slider("Rows", 1, 20, 4)
                c = st.slider("Columns", 1, 20, 5)
                shape = (r, c)
            else:
                d1 = st.slider("Depth", 1, 5, 2)
                d2 = st.slider("Rows", 1, 10, 3)
                d3 = st.slider("Columns", 1, 10, 4)
                shape = (d1, d2, d3)

            dtype_choice = st.selectbox("Data Type (dtype)", ["float64", "float32", "int64", "int32", "int16"])

            # Specific parameters per method
            warning_msg = None
            kw = {}

            if method == "np.empty":
                st.warning("⚠️ `np.empty` creates an uninitialized array containing arbitrary RAM garbage!")

            elif method == "np.arange":
                start_val = st.number_input("Start", value=0.0)
                stop_val = st.number_input("Stop", value=10.0)
                step_val = st.number_input("Step", value=1.0)
                kw = {"start": start_val, "stop": stop_val, "step": step_val}

            elif method == "np.linspace":
                lin_start = st.number_input("Start", value=0.0)
                lin_stop = st.number_input("Stop", value=1.0)
                lin_num = st.number_input("Number of samples (num)", value=10, min_value=2, max_value=200)
                lin_end = st.checkbox("Include endpoint?", value=True)
                kw = {"start": lin_start, "stop": lin_stop, "num": lin_num, "endpoint": lin_end}

            elif method == "np.eye":
                k_val = st.slider("Diagonal Offset (k)", -5, 5, 0)
                kw = {"k": k_val}

            elif method == "np.full":
                fill_val = st.number_input("Fill Value", value=7.0)
                kw = {"fill_value": fill_val}

            elif method == "np.random.default_rng":
                dist = st.selectbox("Distribution", ["uniform", "normal", "integers"])
                seed_val = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
                kw = {"distribution": dist, "seed": seed_val}
                if dist == "uniform":
                    kw["low"] = st.number_input("Low", value=0.0)
                    kw["high"] = st.number_input("High", value=10.0)
                elif dist == "normal":
                    kw["mean"] = st.number_input("Mean (μ)", value=0.0)
                    kw["std"] = st.number_input("Std Dev (σ)", value=1.0, min_value=0.1)
                else:
                    kw["low"] = st.number_input("Low (inclusive)", value=1)
                    kw["high"] = st.number_input("High (exclusive)", value=100)

            elif method == "Custom List/Array":
                custom_str = st.text_area("Python List syntax", value="[[1, 2, 3], [4, 5, 6]]")

        with col_view:
            st.markdown("#### 📋 Array Preview & Properties")
            try:
                if method == "np.zeros":
                    arr = create_zeros(shape, dtype=dtype_choice)
                    code_snippet = generate_creation_code("zeros", shape, dtype_choice)
                elif method == "np.ones":
                    arr = create_ones(shape, dtype=dtype_choice)
                    code_snippet = generate_creation_code("ones", shape, dtype_choice)
                elif method == "np.empty":
                    arr, warning_msg = create_empty(shape, dtype=dtype_choice)
                    code_snippet = generate_creation_code("empty", shape, dtype_choice)
                elif method == "np.arange":
                    arr = create_arange(kw["start"], kw["stop"], kw["step"], dtype=dtype_choice)
                    code_snippet = generate_creation_code("arange", arr.shape, dtype_choice, **kw)
                elif method == "np.linspace":
                    arr = create_linspace(kw["start"], kw["stop"], num=int(kw["num"]), endpoint=kw["endpoint"], dtype=dtype_choice)
                    code_snippet = generate_creation_code("linspace", arr.shape, dtype_choice, **kw)
                elif method == "np.eye":
                    N = shape[0] if isinstance(shape, tuple) else shape
                    M = shape[1] if isinstance(shape, tuple) and len(shape) > 1 else None
                    arr = create_eye(N, M=M, k=kw["k"], dtype=dtype_choice)
                    code_snippet = generate_creation_code("eye", arr.shape, dtype_choice, **kw)
                elif method == "np.full":
                    arr = create_full(shape, fill_value=kw["fill_value"], dtype=dtype_choice)
                    code_snippet = generate_creation_code("full", shape, dtype_choice, **kw)
                elif method == "np.random.default_rng":
                    arr = create_random_array(shape, dtype=dtype_choice, **kw)
                    code_snippet = generate_creation_code("random", shape, dtype_choice, **kw)
                else:
                    import ast
                    raw_data = ast.literal_eval(custom_str)
                    arr = create_from_data(raw_data, dtype=dtype_choice)
                    code_snippet = f"import numpy as np\narr = np.array({custom_str}, dtype='{dtype_choice}')"

                if warning_msg:
                    st.warning(warning_msg)

                # Properties bar
                props = get_array_properties(arr)
                render_properties_bar(props)

                # Array visualization
                if arr.ndim == 1:
                    st.plotly_chart(plot_1d_array(arr, title=f"1D Array ({len(arr)} elements)"), use_container_width=True)
                elif arr.ndim == 2:
                    st.plotly_chart(plot_2d_heatmap(arr, title="2D Matrix Heatmap"), use_container_width=True)
                else:
                    st.info(f"3D Tensor with shape {arr.shape}. Showing 2D slice at depth index [0, :, :]:")
                    st.plotly_chart(plot_2d_heatmap(arr[0], title=f"Tensor Slice [0, :, :] (Shape: {arr[0].shape})"), use_container_width=True)

                st.markdown("#### 💻 Equivalent NumPy Code")
                st.code(code_snippet, language="python")

            except Exception as e:
                st.error(f"Error during array creation: {str(e)}")

    with tabs[1]:
        st.markdown(
            """
            ### 📘 Concept Guide: NumPy Array Constructors
            - **`np.zeros(shape)` / `np.ones(shape)`**: Allocates and explicitly populates memory with 0s or 1s.
            - **`np.empty(shape)`**: Blazingly fast memory allocation because it skips initialization. Use only when you intend to overwrite every single element immediately!
            - **`np.arange(start, stop, step)`**: Like Python `range()`, but returns an ndarray and supports floating-point increments.
            - **`np.linspace(start, stop, num)`**: When you need exact sample counts over an interval (indispensable for plotting functions and mathematical domains).
            - **`np.random.default_rng(seed)`**: Modern NumPy recommended random API (replaces deprecated `np.random.seed` and `np.random.rand`).
            
            ### ⚠️ Common Beginner Mistakes:
            1. **Calling `np.zeros(3, 4)` instead of `np.zeros((3, 4))`**: Shape must be passed as a tuple!
            2. **Reading from `np.empty`**: Forgetting that `np.empty` contains uninitialized garbage data.
            3. **Floating point step error in `np.arange`**: Due to IEEE 754 precision issues, `np.arange(0, 1.0, 0.1)` might include or exclude the endpoint unpredictably. Prefer `np.linspace` for fractional steps.
            
            ### 🏢 Real-World Use Case:
            Initializing neural network weight tensors with reproducible Gaussian random distributions (`mean=0.0`, `std=0.01`, fixed random seed).
            """
        )


# ==========================================
# PAGE 2: INDEXING AND SLICING LAB
# ==========================================
elif selected_page == "🔍 2. Indexing & Slicing Lab":
    st.markdown("<div class='main-title'>🔍 Indexing & Slicing Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Master element selection, slicing syntax, boolean masking, and Views vs Copies</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Slicing", "💥 View vs Copy Mutator", "Learning Guide"])

    # Baseline 2D array for indexing
    rng_idx = np.random.default_rng(42)
    default_matrix = rng_idx.integers(10, 99, size=(6, 8))

    with tabs[0]:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### 🎯 Selection Mode")
            mode = st.selectbox(
                "Select Indexing Strategy",
                [
                    "Single Element [row, col]",
                    "Row or Column Extraction",
                    "2D Submatrix Slicing [r_start:r_stop:step, c_start:c_stop:step]",
                    "Boolean Mask Filtering",
                    "Fancy Indexing (Integer List)",
                    "Diagonal Extraction",
                    "Reverse Array [::-1]"
                ]
            )

            if mode == "Single Element [row, col]":
                row_idx = st.number_input("Row index (supports negative)", value=1, min_value=-6, max_value=5)
                col_idx = st.number_input("Col index (supports negative)", value=2, min_value=-8, max_value=7)

            elif mode == "Row or Column Extraction":
                axis_pick = st.radio("Extract", ["Row (axis 0)", "Column (axis 1)"])
                pick_idx = st.number_input("Index", value=0, min_value=-6, max_value=7)

            elif mode == "2D Submatrix Slicing [r_start:r_stop:step, c_start:c_stop:step]":
                st.caption("Row slice boundaries:")
                r_start = st.number_input("Row Start", value=1, min_value=0, max_value=6)
                r_stop = st.number_input("Row Stop", value=5, min_value=0, max_value=6)
                r_step = st.number_input("Row Step", value=1, min_value=1, max_value=4)

                st.caption("Column slice boundaries:")
                c_start = st.number_input("Col Start", value=2, min_value=0, max_value=8)
                c_stop = st.number_input("Col Stop", value=7, min_value=0, max_value=8)
                c_step = st.number_input("Col Step", value=2, min_value=1, max_value=4)

            elif mode == "Boolean Mask Filtering":
                cond = st.selectbox("Condition", [">", "<", ">=", "<=", "==", "!=", "even", "odd", "between"])
                thresh = st.number_input("Threshold", value=50.0)
                upper = st.number_input("Upper Bound (for 'between')", value=75.0)

            elif mode == "Fancy Indexing (Integer List)":
                fancy_axis = st.radio("Fancy Index Axis", ["Rows (axis 0)", "Columns (axis 1)"])
                fancy_str = st.text_input("Indices list (comma-separated)", value="0, 2, 4")

            elif mode == "Diagonal Extraction":
                diag_k = st.slider("Diagonal Offset (k)", -4, 4, 0)

        with col2:
            st.markdown("#### 📊 Visual Result & Active Mask")
            try:
                if mode == "Single Element [row, col]":
                    val, code = safe_index_2d(default_matrix, row_idx, col_idx)
                    st.success(f"**Extracted Element:** `{val}`")
                    st.code(code, language="python")
                    # Show on heatmap
                    r_slc = slice(row_idx, row_idx + 1 if row_idx != -1 else 6)
                    c_slc = slice(col_idx, col_idx + 1 if col_idx != -1 else 8)
                    st.plotly_chart(plot_sliced_submatrix(default_matrix, r_slc, c_slc, title=f"Selected Cell [{row_idx}, {col_idx}]"), use_container_width=True)

                elif mode == "Row or Column Extraction":
                    axis_num = 0 if "Row" in axis_pick else 1
                    res, code = select_row_or_col(default_matrix, pick_idx, axis=axis_num)
                    st.success(f"**Extracted Vector (Length {len(res)}):** `{res.tolist()}`")
                    st.code(code, language="python")
                    r_slc = slice(pick_idx, pick_idx + 1) if axis_num == 0 else slice(None)
                    c_slc = slice(pick_idx, pick_idx + 1) if axis_num == 1 else slice(None)
                    st.plotly_chart(plot_sliced_submatrix(default_matrix, r_slc, c_slc, title=code), use_container_width=True)

                elif mode == "2D Submatrix Slicing [r_start:r_stop:step, c_start:c_stop:step]":
                    submat, code = slice_2d_submatrix(default_matrix, r_start, r_stop, r_step, c_start, c_stop, c_step)
                    st.success(f"**Submatrix Shape:** `{submat.shape}`")
                    st.code(code, language="python")
                    st.plotly_chart(plot_sliced_submatrix(default_matrix, slice(r_start, r_stop, r_step), slice(c_start, c_stop, c_step), title=code), use_container_width=True)

                elif mode == "Boolean Mask Filtering":
                    filtered, mask, code = filter_boolean(default_matrix, cond, thresh, upper)
                    st.success(f"**Matching Elements Found:** `{len(filtered)}` elements")
                    st.code(code, language="python")
                    st.write(f"**Filtered 1D Result:** `{filtered.tolist()}`")
                    # Visual mask
                    st.plotly_chart(plot_2d_heatmap(mask.astype(int), title="Boolean Mask (1 = True, 0 = False)", colorscale="Blues"), use_container_width=True)

                elif mode == "Fancy Indexing (Integer List)":
                    idx_list = [int(x.strip()) for x in fancy_str.split(",") if x.strip()]
                    ax_num = 0 if "Rows" in fancy_axis else 1
                    fancy_res, code = fancy_indexing(default_matrix, idx_list, axis=ax_num)
                    st.success(f"**Result Shape:** `{fancy_res.shape}`")
                    st.code(code, language="python")
                    st.plotly_chart(plot_2d_heatmap(fancy_res, title=f"Fancy Indexed Matrix ({fancy_res.shape})"), use_container_width=True)

                elif mode == "Diagonal Extraction":
                    diag, code = extract_diagonal(default_matrix, offset=diag_k)
                    st.success(f"**Diagonal Elements (k={diag_k}):** `{diag.tolist()}`")
                    st.code(code, language="python")
                    st.plotly_chart(plot_1d_array(diag, title=f"Diagonal Vector (k={diag_k})"), use_container_width=True)

                elif mode == "Reverse Array [::-1]":
                    rev, code = reverse_array(default_matrix)
                    st.success(f"**Reversed Array Shape:** `{rev.shape}`")
                    st.code(code, language="python")
                    st.plotly_chart(plot_2d_heatmap(rev, title="Reversed Matrix (Rows & Columns Flipped)"), use_container_width=True)

            except Exception as e:
                st.error(f"Indexing Error: {str(e)}")

    with tabs[1]:
        st.markdown("### 💥 The Critical 'View vs Copy' Mutation Experiment")
        st.markdown(
            """
            In Python lists, slicing `list[0:2]` always creates a shallow copy.
            In NumPy, **basic slicing `arr[0:2]` creates a VIEW**!
            A view shares the exact same memory buffer. If you modify a view, you **accidentally mutate your original array**!
            """
        )

        mut_val = st.number_input("Mutation Value to inject into slice", value=999.0)
        base_test = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        col_v1, col_v2 = st.columns(2)
        demo_res = demonstrate_view_vs_copy(base_test, slice_start=0, slice_stop=2, mutation_value=mut_val)

        with col_v1:
            st.error("🚨 Scenario A: Basic Slice (VIEW)")
            st.markdown(f"**Initial Array:** `{demo_res['original_initial'].tolist()}`")
            st.markdown("```python\nview = arr[0:2]\nview[:] = 999.0  # Mutating the slice\n```")
            st.markdown(f"**Original Array After Mutation:**")
            st.write(demo_res["view_modified_original"])
            st.caption(demo_res["view_explanation"])

        with col_v2:
            st.success("🛡️ Scenario B: Protected Copy (`.copy()`)")
            st.markdown(f"**Initial Array:** `{demo_res['original_initial'].tolist()}`")
            st.markdown("```python\nsafe_copy = arr[0:2].copy()\nsafe_copy[:] = 999.0  # Mutating the copy\n```")
            st.markdown(f"**Original Array After Mutation:**")
            st.write(demo_res["copy_modified_original"])
            st.caption(demo_res["copy_explanation"])

    with tabs[2]:
        st.markdown(
            """
            ### 📘 Concept Guide: Slicing Rules & Memory
            - **Basic Slicing Syntax**: `arr[start:stop:step]`. Negative start/stop counts backward from the end.
            - **View vs Copy Rule of Thumb**:
              - Basic Slicing (`arr[1:4]`) returns a **VIEW**.
              - Fancy Indexing (`arr[[0, 2]]`) and Boolean Masking (`arr[arr > 5]`) always allocate a **COPY**.
            
            ### ⚠️ Common Pitfalls:
            1. **Unintended In-place Mutation**: Mutating a slice without `.copy()` leading to silent bugs in data pipelines.
            2. **Chained Indexing**: `arr[0][1]` creates temporary Python objects. Always use multi-dimensional index tuple `arr[0, 1]`.
            """
        )


# ==========================================
# PAGE 3: MATHEMATICAL OPERATIONS
# ==========================================
elif selected_page == "➕ 3. Mathematical Operations":
    st.markdown("<div class='main-title'>➕ Mathematical Operations</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Element-wise arithmetic, powers, logarithms, trigonometric, and conditional logic</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Operation Playground", "Learning Guide"])

    with tabs[0]:
        col_ctrl, col_res = st.columns([1, 2])

        with col_ctrl:
            category = st.selectbox("Operation Category", ["Binary Arithmetic (A and B)", "Unary Mathematical Functions (f(A))", "Comparisons & np.where"])

            # Input Matrix A
            st.markdown("##### Matrix A Config")
            a_preset = st.selectbox("Preset for Matrix A", ["Positive Integers (3x3)", "Uniform Floats (3x3)", "With Negatives and Zeros (3x3)"])
            if a_preset == "Positive Integers (3x3)":
                arr_a = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
            elif a_preset == "Uniform Floats (3x3)":
                arr_a = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])
            else:
                arr_a = np.array([[-4.0, 0.0, 9.0], [16.0, -25.0, 36.0], [0.0, 49.0, -64.0]])

            if category == "Binary Arithmetic (A and B)":
                op_name = st.selectbox("Operation", ["add (+)", "subtract (-)", "multiply (*)", "divide (/)", "floor_divide (//)", "mod (%)", "power (**)"])
                op_key = op_name.split()[0]
                
                b_type = st.radio("Operand B Type", ["Scalar", "Matrix B (3x3)"])
                if b_type == "Scalar":
                    b_val = st.number_input("Scalar B value", value=2.0)
                    operand_b = b_val
                else:
                    b_preset = st.selectbox("Matrix B Preset", ["All Twos", "Contains Zeros", "Sequential [1..9]"])
                    if b_preset == "All Twos":
                        operand_b = np.full((3, 3), 2.0)
                    elif b_preset == "Contains Zeros":
                        operand_b = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 2.0], [5.0, 0.0, 3.0]])
                    else:
                        operand_b = np.arange(1, 10, dtype=float).reshape(3, 3)

            elif category == "Unary Mathematical Functions (f(A))":
                unary_op = st.selectbox("Function", ["sqrt", "abs", "exp", "log", "log10", "log2", "round", "floor", "ceil", "sin", "cos"])
                dec_places = st.slider("Decimal places (for round)", 0, 4, 2)

            else:  # Comparisons & np.where
                comp_op = st.selectbox("Comparison Operator", [">", "<", ">=", "<=", "==", "!="])
                comp_val = st.number_input("Compare Against", value=25.0)
                st.caption("np.where replacement values:")
                where_x = st.number_input("Value if True (x)", value=100.0)
                where_y = st.number_input("Value if False (y)", value=-1.0)

        with col_res:
            st.markdown("#### 📋 Calculation Result")
            try:
                if category == "Binary Arithmetic (A and B)":
                    res, code, expl, warn = compute_arithmetic(arr_a, operand_b, op_key)
                    if warn:
                        st.warning(f"⚠️ {warn}")
                    st.code(code, language="python")
                    st.info(expl)
                    st.plotly_chart(plot_2d_heatmap(res, title="Computation Result Matrix"), use_container_width=True)

                elif category == "Unary Mathematical Functions (f(A))":
                    res, code, expl, warn = compute_unary_math(arr_a, unary_op, decimals=dec_places)
                    if warn:
                        st.warning(f"⚠️ {warn}")
                    st.code(code, language="python")
                    st.info(expl)
                    st.plotly_chart(plot_2d_heatmap(res, title=f"Result of np.{unary_op}(A)"), use_container_width=True)

                else:
                    comp_mask, comp_code, comp_expl = compute_comparison(arr_a, comp_val, comp_op)
                    where_res, where_code, where_expl = compute_where(comp_mask, where_x, where_y)
                    
                    st.markdown("##### 1. Comparison Mask")
                    st.code(comp_code, language="python")
                    st.caption(comp_expl)
                    
                    st.markdown("##### 2. np.where Vectorized Conditional Output")
                    st.code(f"{comp_code}\nresult = np.where(mask, {where_x}, {where_y})", language="python")
                    st.caption(where_expl)
                    
                    st.plotly_chart(plot_2d_heatmap(where_res, title=f"np.where(A {comp_op} {comp_val}, {where_x}, {where_y})"), use_container_width=True)

            except Exception as e:
                st.error(f"Mathematical execution error: {str(e)}")

    with tabs[1]:
        st.markdown(
            """
            ### 📘 Concept Guide: Vectorized Arithmetic & Edge Cases
            - **No Python Loops Required**: `a + b` executes in optimized C/Fortran routines with hardware acceleration.
            - **IEEE 754 Floating-Point Standard**:
              - `1.0 / 0.0` -> `inf` (Infinity)
              - `-1.0 / 0.0` -> `-inf`
              - `0.0 / 0.0` -> `nan` (Not a Number)
              - `np.sqrt(-1.0)` -> `nan`
              - `np.log(0.0)` -> `-inf`
            - **Vectorized Conditionals (`np.where`)**:
              Replaces slow `if/else` loops across matrices.
              `np.where(condition, x, y)` chooses `x` when True, `y` when False.
            """
        )


# ==========================================
# PAGE 4: STATISTICAL & AXIS LAB
# ==========================================
elif selected_page == "📊 4. Statistical & Axis Lab":
    st.markdown("<div class='main-title'>📊 Statistical & Axis Operations</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Master NumPy statistical metrics, axis mechanics (0 vs 1 vs None), and degrees of freedom</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Analytics", "Axis Mechanics Deep Dive", "Learning Guide"])

    # Load synthetic dataset
    sample_datasets = generate_sample_datasets()

    with tabs[0]:
        col1, col2 = st.columns([1, 2])

        with col1:
            dataset_choice = st.selectbox("Select Analytical Dataset", list(sample_datasets.keys()))
            df_curr, arr_curr, desc_curr = sample_datasets[dataset_choice]
            st.caption(desc_curr)

            axis_selection = st.radio(
                "Aggregation Axis",
                ["None (Global Aggregate)", "0 (Down rows -> Column-wise)", "1 (Across columns -> Row-wise)"]
            )
            axis_param = None if "None" in axis_selection else (0 if "0" in axis_selection else 1)

            ddof_val = st.radio("Degrees of Freedom (ddof)", [0, 1], index=0, help="0=Population (NumPy default), 1=Sample (Bessel's correction / Pandas default)")
            p_q = st.slider("Percentile Rank (q)", 0.0, 100.0, 75.0, 5.0)

        with col2:
            st.markdown(f"#### 📈 Summary Statistics for: {dataset_choice}")
            stats = compute_all_statistics(arr_curr, axis=axis_param, ddof=ddof_val, percentile_q=p_q)

            if axis_param is None:
                # Global scalar stats
                st.markdown(
                    f"""
                    | Metric | Value | Metric | Value |
                    | :--- | :--- | :--- | :--- |
                    | **Sum** | `{stats['sum']:,.2f}` | **Mean (μ)** | `{stats['mean']:,.2f}` |
                    | **Median** | `{stats['median']:,.2f}` | **Std Dev (σ)** | `{stats['std']:,.2f}` (ddof={ddof_val}) |
                    | **Variance** | `{stats['var']:,.2f}` | **Range (PTP)** | `{stats['ptp']:,.2f}` |
                    | **Min** | `{stats['min']:,.2f}` (idx: `{stats['argmin']}`) | **Max** | `{stats['max']:,.2f}` (idx: `{stats['argmax']}`) |
                    | **25th Quantile** | `{stats['quantile_25']:,.2f}` | **{p_q}th Percentile** | `{stats['percentile']:,.2f}` |
                    """
                )
                st.plotly_chart(plot_distribution_analysis(arr_curr, title=f"Distribution of {dataset_choice}"), use_container_width=True)

            else:
                # Axis-wise vectors
                labels = df_curr.columns.tolist() if axis_param == 0 else df_curr.index.tolist()
                stat_df = pd.DataFrame({
                    "Mean": stats["mean"],
                    "Median": stats["median"],
                    "Std Dev": stats["std"],
                    "Min": stats["min"],
                    "Max": stats["max"],
                    f"P{int(p_q)}": stats["percentile"]
                }, index=labels)

                st.dataframe(stat_df.round(2), use_container_width=True)

                # Axis chart
                bar_fig = px.bar(
                    stat_df,
                    barmode="group",
                    title=f"Axis {axis_param} Metrics by {'Column' if axis_param==0 else 'Row'}",
                    labels={"value": "Magnitude", "index": "Entity"}
                )
                st.plotly_chart(bar_fig, use_container_width=True)

    with tabs[1]:
        st.markdown("### 🧭 Axis Mechanics Demystified")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### `axis=0` (The Vertical Collapse)")
            st.markdown(
                """
                - Aggregates **down the rows** (along index direction 0).
                - Collapses the row dimension $M$, leaving $N$ outputs.
                - **Mental Shortcut:** For a table of student marks, `np.mean(marks, axis=0)` gives the **average score per subject**.
                """
            )
            st.code("col_means = np.mean(arr, axis=0)  # Shape (N,)", language="python")

        with col_b:
            st.markdown("#### `axis=1` (The Horizontal Collapse)")
            st.markdown(
                """
                - Aggregates **across the columns** (along column direction 1).
                - Collapses the column dimension $N$, leaving $M$ outputs.
                - **Mental Shortcut:** For a table of student marks, `np.mean(marks, axis=1)` gives the **GPA / overall mark for each student**.
                """
            )
            st.code("row_means = np.mean(arr, axis=1)  # Shape (M,)", language="python")

        st.divider()
        st.markdown("#### ⚖️ `ddof=0` vs `ddof=1` Explained")
        st.info(get_ddof_explanation(ddof_val))

    with tabs[2]:
        st.markdown(
            """
            ### 📘 Concept Guide: NumPy Statistics
            - **`np.sum` / `np.prod`**: Vectorized addition or multiplication across elements.
            - **`np.cumsum` / `np.cumprod`**: Cumulative running totals (essential for financial portfolios and time series).
            - **`np.argmin` / `np.argmax`**: Return index of the minimum or maximum element.
            - **Percentile vs Quantile**: `np.percentile(arr, 50)` takes inputs from `0..100`; `np.quantile(arr, 0.5)` takes inputs from `0.0..1.0`.
            """
        )


# ==========================================
# PAGE 5: RESHAPING & MANIPULATION
# ==========================================
elif selected_page == "🔄 5. Reshaping & Manipulation":
    st.markdown("<div class='main-title'>🔄 Reshaping & Array Manipulation</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Reshape, flatten, ravel, transpose, concatenate, stack, split, expand_dims, and tile</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Playground", "Learning Guide"])

    # Base test array (12 elements)
    base_arr = np.arange(12)

    with tabs[0]:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### ⚙️ Manipulation Selection")
            manip_op = st.selectbox(
                "Operation",
                [
                    "reshape()",
                    "ravel() vs flatten() (View vs Copy)",
                    "transpose() / .T",
                    "concatenate() / vstack() / hstack()",
                    "split() / array_split()",
                    "expand_dims() / squeeze()",
                    "repeat() / tile()",
                    "sort() / unique()"
                ]
            )

            if manip_op == "reshape()":
                st.caption(f"Original Array Size: {base_arr.size} elements.")
                r_dim = st.number_input("Target Rows", value=3, min_value=-1, max_value=12)
                c_dim = st.number_input("Target Columns", value=4, min_value=-1, max_value=12)
                target_shape = (r_dim, c_dim)

            elif manip_op == "ravel() vs flatten() (View vs Copy)":
                matrix_sample = base_arr.reshape(3, 4)
                flatten_method = st.radio("Method", ["ravel", "flatten"])

            elif manip_op == "transpose() / .T":
                matrix_sample = base_arr.reshape(3, 4)

            elif manip_op == "concatenate() / vstack() / hstack()":
                join_method = st.selectbox("Join Function", ["concatenate", "vstack", "hstack", "stack"])
                join_axis = st.selectbox("Axis", [0, 1])

            elif manip_op == "split() / array_split()":
                sections = st.slider("Number of sections", 1, 6, 3)
                allow_unequal = st.checkbox("Allow unequal splits (array_split)?", value=False)

            elif manip_op == "expand_dims() / squeeze()":
                dim_choice = st.radio("Action", ["expand_dims", "squeeze"])
                dim_axis = st.slider("Axis", 0, 2, 0)

            elif manip_op == "repeat() / tile()":
                rep_action = st.radio("Action", ["repeat", "tile"])
                rep_count = st.slider("Multiplier", 2, 4, 2)

            elif manip_op == "sort() / unique()":
                sort_action = st.radio("Action", ["sort", "unique"])

        with col2:
            st.markdown("#### 📋 Operation Result")
            try:
                if manip_op == "reshape()":
                    is_valid, val_msg = validate_reshape(base_arr.size, target_shape)
                    if not is_valid:
                        st.error(f"❌ Shape Incompatibility: {val_msg}")
                    else:
                        st.success(f"✅ {val_msg}")
                        reshaped, is_view, code = execute_reshape(base_arr, target_shape)
                        st.code(code, language="python")
                        st.markdown(f"**Returns View?** {'✅ YES (Shares Memory)' if is_view else '❌ NO (Allocated Copy)'}")
                        st.plotly_chart(plot_2d_heatmap(reshaped, title=f"Reshaped Matrix {target_shape}"), use_container_width=True)

                elif manip_op == "ravel() vs flatten() (View vs Copy)":
                    flat_res, is_view, code, note = execute_flatten_vs_ravel(matrix_sample, flatten_method)
                    st.code(code, language="python")
                    tag_class = "badge-view" if is_view else "badge-copy"
                    st.markdown(f"<span class='badge {tag_class}'>{'VIEW' if is_view else 'COPY'}</span>", unsafe_allow_html=True)
                    st.info(note)
                    st.plotly_chart(plot_1d_array(flat_res, title=f"{flatten_method}() Result"), use_container_width=True)

                elif manip_op == "transpose() / .T":
                    trans, is_view, code = execute_transpose(matrix_sample)
                    st.code(code, language="python")
                    st.plotly_chart(plot_2d_heatmap(trans, title=f"Transposed Matrix (Original: {matrix_sample.shape} -> New: {trans.shape})"), use_container_width=True)

                elif manip_op == "concatenate() / vstack() / hstack()":
                    arr1 = np.ones((2, 3), dtype=int) * 1
                    arr2 = np.ones((2, 3), dtype=int) * 2
                    res_join, code = execute_join_operation(arr1, arr2, operation=join_method, axis=join_axis)
                    st.code(code, language="python")
                    st.success(f"**Joined Array Shape:** `{res_join.shape}`")
                    st.plotly_chart(plot_2d_heatmap(res_join, title=f"{join_method} Result ({res_join.shape})"), use_container_width=True)

                elif manip_op == "split() / array_split()":
                    mat_split = np.arange(12).reshape(4, 3)
                    splits, code = execute_split_operation(mat_split, sections=sections, axis=0, allow_unequal=allow_unequal)
                    st.code(code, language="python")
                    st.success(f"Successfully split into {len(splits)} sub-arrays!")
                    for i, sp in enumerate(splits):
                        st.write(f"**Chunk {i+1} (Shape {sp.shape}):**")
                        st.dataframe(sp, use_container_width=True)

                elif manip_op == "expand_dims() / squeeze()":
                    base_1d = np.array([10, 20, 30])
                    res_exp, is_view, code = execute_expand_or_squeeze(base_1d, dim_choice, axis=dim_axis)
                    st.code(code, language="python")
                    st.success(f"**Original Shape:** `{base_1d.shape}` ➔ **New Shape:** `{res_exp.shape}`")

                elif manip_op == "repeat() / tile()":
                    base_seq = np.array([1, 2, 3])
                    rep_res, code = execute_repeat_or_tile(base_seq, rep_action, rep_count)
                    st.code(code, language="python")
                    st.success(f"**Result:** `{rep_res.tolist()}`")
                    st.plotly_chart(plot_1d_array(rep_res, title=f"{rep_action}(repeats={rep_count})"), use_container_width=True)

                elif manip_op == "sort() / unique()":
                    noisy_arr = np.array([42, 10, 7, 42, 99, 10, 5, 7, 100])
                    st.markdown(f"**Input Array:** `{noisy_arr.tolist()}`")
                    sorted_res, counts, code = execute_sort_or_unique(noisy_arr, sort_action)
                    st.code(code, language="python")
                    if sort_action == "unique":
                        st.write(f"**Distinct Elements:** `{sorted_res.tolist()}`")
                        st.write(f"**Occurrences Counts:** `{counts.tolist()}`")
                    else:
                        st.write(f"**Ascending Sort:** `{sorted_res.tolist()}`")

            except Exception as e:
                st.error(f"Manipulation error: {str(e)}")

    with tabs[1]:
        st.markdown(
            """
            ### 📘 Concept Guide: Reshaping & Views
            - **`reshape(shape)`**: Changes shape without changing memory data, provided `prod(new_shape) == arr.size`.
            - **`-1` Dimension Inference**: NumPy can automatically deduce one unknown dimension: `arr.reshape(3, -1)`.
            - **`ravel()` vs `flatten()`**:
              - `ravel()` returns a **view** when the array is C-contiguous.
              - `flatten()` **always allocates a copy**.
            - **`concatenate` vs `stack`**:
              - `concatenate` joins along an **existing axis**.
              - `stack` joins along a **new axis**, increasing the array rank by 1.
            """
        )


# ==========================================
# PAGE 6: BROADCASTING VISUALIZER
# ==========================================
elif selected_page == "📡 6. Broadcasting Visualizer":
    st.markdown("<div class='main-title'>📡 Broadcasting Visualizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Interactive step-by-step diagnostic engine for NumPy's 4 broadcasting rules</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Rule Checker", "Practical Broadcasting Scenarios", "Learning Guide"])

    with tabs[0]:
        st.markdown("#### 🔍 Custom Shape Compatibility Checker")
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("##### Array A Dimensions")
            shape_a_str = st.text_input("Shape of Operand A (comma-separated)", value="3, 1")
        with col_s2:
            st.markdown("##### Array B Dimensions")
            shape_b_str = st.text_input("Shape of Operand B (comma-separated)", value="1, 4")

        try:
            shape_a = tuple(int(x.strip()) for x in shape_a_str.split(",") if x.strip())
            shape_b = tuple(int(x.strip()) for x in shape_b_str.split(",") if x.strip())

            analysis = analyze_broadcasting(shape_a, shape_b)

            if analysis["is_compatible"]:
                st.success(f"✅ **Broadcasting COMPATIBLE!** Output shape will be: `{analysis['result_shape']}`")
            else:
                st.error(f"❌ **INCOMPATIBLE SHAPES!** {analysis['failure_reason']}")

            st.markdown("##### 📐 Right-to-Left Alignment Table")
            step_df = pd.DataFrame(analysis["steps"])
            st.dataframe(step_df, use_container_width=True)

            # Interactive execution if small enough
            if analysis["is_compatible"] and np.prod(analysis["result_shape"]) <= 50:
                rng_b = np.random.default_rng(42)
                arr_a_live = rng_b.integers(1, 10, size=shape_a)
                arr_b_live = rng_b.integers(10, 50, size=shape_b)
                res_b, final_s, code_b = broadcast_and_operate(arr_a_live, arr_b_live, "add")

                st.markdown("##### 💻 Live Execution:")
                st.code(code_b, language="python")
                st.write(f"**Operand A (Shape {shape_a}):**")
                st.write(arr_a_live)
                st.write(f"**Operand B (Shape {shape_b}):**")
                st.write(arr_b_live)
                st.write(f"**Broadcast Result A + B (Shape {final_s}):**")
                st.write(res_b)

        except Exception as e:
            st.error(f"Broadcasting evaluation error: {str(e)}")

    with tabs[1]:
        scenarios = create_broadcasting_scenarios()
        chosen_scen = st.selectbox("Select Practical Scenario", list(scenarios.keys()))
        info = scenarios[chosen_scen]

        st.info(f"**Description:** {info['description']}\n\n**Concept:** {info['concept']}")
        scen_a = info["a"]
        scen_b = info["b"]

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Operand A (Shape: {scen_a.shape}):**")
            st.write(scen_a)
        with col2:
            st.write(f"**Operand B (Shape: {scen_b.shape}):**")
            st.write(scen_b)

        try:
            scen_res, scen_shape, scen_code = broadcast_and_operate(scen_a, scen_b, "add")
            st.success(f"**Broadcast Result (Shape: {scen_shape}):**")
            st.write(scen_res)
            st.code(scen_code, language="python")
        except ValueError as err:
            st.error(f"Broadcasting Conflict (Expected Demo Error):\n{str(err)}")

    with tabs[2]:
        st.markdown(
            """
            ### 📘 The 4 Golden Rules of NumPy Broadcasting
            Broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations without copying data.
            
            1. **Rule 1 (Right-to-Left Alignment):**
               Dimensions are compared starting from the **trailing (rightmost) dimension** and moving backward to the leading dimension.
            2. **Rule 2 (Left-Padding):**
               If one array has fewer dimensions, it is prepended with 1s on the left until both ranks match.
            3. **Rule 3 (Compatibility Criteria):**
               Two dimensions are compatible if:
               - They are equal, **OR**
               - One of them is **1**.
            4. **Rule 4 (Stretch):**
               A dimension of size 1 is conceptually stretched to match the larger dimension without copying memory.
            """
        )


# ==========================================
# PAGE 7: NUMPY FILE OPERATIONS
# ==========================================
elif selected_page == "💾 7. NumPy File Operations":
    st.markdown("<div class='main-title'>💾 NumPy File Operations</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Serialize, save, download, and securely load .npy, .npz, and .csv files with allow_pickle=False</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Save & Export", "Upload & Inspect", "Storage Format Comparison"])

    # Sample array for saving
    save_sample = np.arange(24, dtype=float).reshape(4, 6)

    with tabs[0]:
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.markdown("#### 📤 Export Configuration")
            export_fmt = st.selectbox("Export Format", [".npy (Single Array Binary)", ".npz (Multi-Array Archive)", ".csv / .txt (Plaintext)"])
            compress_npz = st.checkbox("Compress Archive (np.savez_compressed)?", value=False) if ".npz" in export_fmt else False

        with col_s2:
            st.markdown("#### 💾 Direct In-Memory Browser Download")
            st.write(f"Array to export (Shape: {save_sample.shape}, Dtype: {save_sample.dtype}):")
            st.dataframe(save_sample, use_container_width=True)

            if ".npy" in export_fmt:
                npy_bytes = serialize_to_npy(save_sample)
                st.download_button(
                    label="⬇️ Download array.npy",
                    data=npy_bytes,
                    file_name="matrix_data.npy",
                    mime="application/octet-stream"
                )
                st.code("np.save('matrix_data.npy', arr, allow_pickle=False)", language="python")

            elif ".npz" in export_fmt:
                archive_dict = {
                    "matrix_main": save_sample,
                    "row_totals": np.sum(save_sample, axis=1),
                    "col_totals": np.sum(save_sample, axis=0)
                }
                npz_bytes = serialize_to_npz(archive_dict, compressed=compress_npz)
                st.download_button(
                    label="⬇️ Download dataset.npz (Contains 3 arrays)",
                    data=npz_bytes,
                    file_name="dataset_archive.npz",
                    mime="application/zip"
                )
                st.code(f"np.{'savez_compressed' if compress_npz else 'savez'}('dataset.npz', **dict_of_arrays)", language="python")

            else:
                csv_str = serialize_to_csv_txt(save_sample, delimiter=",", fmt="%.2f")
                st.download_button(
                    label="⬇️ Download data.csv",
                    data=csv_str,
                    file_name="matrix_export.csv",
                    mime="text/csv"
                )
                st.code("np.savetxt('data.csv', arr, delimiter=',', fmt='%.2f')", language="python")

    with tabs[1]:
        st.markdown("#### 📥 Secure Array Deserializer")
        st.caption("Upload your own `.npy`, `.npz`, `.csv`, or `.txt` file to inspect. Strictly verified with `allow_pickle=False`.")

        uploaded_file = st.file_uploader("Upload numerical array file", type=["npy", "npz", "csv", "txt"])
        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.read()
                result_load = deserialize_from_bytes(bytes_data, uploaded_file.name)
                st.success(f"✅ {result_load['summary']}")

                if result_load["is_multi"]:
                    chosen_key = st.selectbox("Select array from archive", result_load["keys"])
                    target_arr = result_load["arrays"][chosen_key]
                else:
                    target_arr = result_load["primary_array"]

                render_properties_bar(get_array_properties(target_arr))

                if target_arr.ndim == 2:
                    st.plotly_chart(plot_2d_heatmap(target_arr, title=f"Uploaded Array: {uploaded_file.name}"), use_container_width=True)
                elif target_arr.ndim == 1:
                    st.plotly_chart(plot_1d_array(target_arr, title=f"Uploaded Array: {uploaded_file.name}"), use_container_width=True)
                else:
                    st.write(target_arr)

            except Exception as e:
                st.error(f"File loading error: {str(e)}")

    with tabs[2]:
        st.markdown("### 📊 Storage Format Matrix")
        comp_df = pd.DataFrame(get_storage_format_comparison())
        st.dataframe(comp_df, use_container_width=True)


# ==========================================
# PAGE 8: PERFORMANCE BENCHMARK
# ==========================================
elif selected_page == "⚡ 8. Performance Benchmark":
    st.markdown("<div class='main-title'>⚡ NumPy vs Python Lists Benchmark</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Empirical micro-benchmarking with warmups, median timing, speedup ratios, and RAM consumption</div>", unsafe_allow_html=True)

    tabs = st.tabs(["Interactive Benchmark Suite", "Scaling Curves (N)", "Memory Footprint Analysis", "Why Vectorization Wins"])

    with tabs[0]:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### ⚙️ Benchmark Parameters")
            bench_op = st.selectbox(
                "Benchmark Operation",
                [
                    "addition",
                    "multiplication",
                    "sum",
                    "scalar_mult",
                    "creation"
                ],
                format_func=lambda x: {
                    "addition": "Element-wise Addition (a + b)",
                    "multiplication": "Element-wise Multiplication (a * b)",
                    "sum": "Global Aggregation sum() vs np.sum()",
                    "scalar_mult": "Scalar Multiplication (a * 2.5)",
                    "creation": "Allocation & Creation"
                }[x]
            )

            bench_size = st.select_slider(
                "Dataset Size (N elements)",
                options=[1_000, 10_000, 100_000, 500_000, 1_000_000],
                value=100_000
            )

            run_btn = st.button("🚀 Run Live Benchmark", type="primary", use_container_width=True)

        with col2:
            if run_btn:
                with st.spinner(f"Executing repeated warmups & benchmark runs for {bench_size:,} elements..."):
                    res_b = benchmark_operation(bench_op, size=bench_size, warmups=2, repeats=5)
                    st.plotly_chart(plot_benchmark_comparison(res_b), use_container_width=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Python List Time", f"{res_b['list_time_ms']:.3f} ms")
                    c2.metric("NumPy Array Time", f"{res_b['numpy_time_ms']:.3f} ms")
                    c3.metric("Speedup Multiplier", f"{res_b['speedup_ratio']:.1f}x")

                    st.info(f"💡 **Why this happened:** {res_b['why_it_happened']}")
            else:
                st.caption("Click 'Run Live Benchmark' to measure real performance on your CPU.")

    with tabs[1]:
        st.markdown("#### 📈 Multi-Size Scaling Curve")
        st.caption("Evaluates how execution time scales across orders of magnitude ($N = 1,000$ to $N = 500,000$).")

        scaling_op = st.selectbox("Scaling Operation", ["addition", "multiplication", "sum", "scalar_mult"])
        if st.button("Generate Scaling Curve", key="scale_btn"):
            with st.spinner("Computing scaling curve across 4 size tiers..."):
                curve_data = run_scaling_benchmark(scaling_op, sizes=[1_000, 10_000, 100_000, 500_000])
                st.plotly_chart(plot_scaling_curves(curve_data), use_container_width=True)

    with tabs[2]:
        st.markdown("#### 🧠 True RAM Memory Consumption")
        mem_size = st.select_slider("Test Size for Memory", [1_000, 10_000, 100_000, 1_000_000], value=100_000)
        mem_data = measure_memory_footprint(mem_size)
        st.plotly_chart(plot_memory_comparison(mem_data), use_container_width=True)
        st.success(mem_data["explanation"])

    with tabs[3]:
        st.markdown(
            """
            ### 🏗️ Why is NumPy so much faster?
            
            1. **No Type Checking in the Loop**:
               Python lists can hold arbitrary objects (e.g. `[1, "two", 3.0]`). For every single operation, Python must dynamically look up the object type and dispatch methods. NumPy enforces homogeneous `dtype`, allowing compiled loops to run unchecked.
            2. **Contiguous Memory & Cache Locality**:
               A NumPy array is laid out sequentially in RAM. When the CPU fetches one element, hardware prefetchers load the entire cache line (64 bytes) into L1/L2 cache, minimizing expensive memory stalls.
            3. **SIMD Vectorization**:
               Modern CPUs support AVX-512, AVX2, and SSE instructions that operate on multiple 64-bit floats simultaneously in a single clock cycle.
            """
        )


# ==========================================
# PAGE 9: PRACTICAL DATA ANALYSIS
# ==========================================
elif selected_page == "📈 9. Practical Data Analysis":
    st.markdown("<div class='main-title'>📈 Practical Data Analysis Lab</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Real-world analytical workflows: student exam performance, regional sales, and quality auditing</div>", unsafe_allow_html=True)

    datasets = generate_sample_datasets()
    chosen_ds = st.selectbox("Choose Analytical Scenario", list(datasets.keys()))
    df_active, arr_active, desc_active = datasets[chosen_ds]

    st.markdown(f"*{desc_active} (Synthetic Data)*")
    st.dataframe(df_active, use_container_width=True)

    st.divider()
    st.markdown("### 🛠️ Data Analyst Workflows via NumPy")

    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.markdown("#### 1. Column-wise Normalization (Min-Max Scaling)")
        st.caption("Maps values into [0.0, 1.0] per feature for machine learning.")
        # NumPy vectorized normalization
        mins = np.min(arr_active, axis=0)
        maxs = np.max(arr_active, axis=0)
        normalized = (arr_active - mins) / np.maximum((maxs - mins), 1e-6)
        st.code("normalized = (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))", language="python")
        st.plotly_chart(plot_2d_heatmap(normalized, title="Normalized Features [0.0, 1.0]"), use_container_width=True)

    with col_w2:
        st.markdown("#### 2. Conditional Selection & Anomaly Flagging")
        st.caption("Identify rows exceeding threshold marks or sales quotas.")
        thresh_pct = st.slider("Select High-Performer Percentile", 50, 95, 80)
        cutoff = np.percentile(arr_active, thresh_pct)
        high_mask = arr_active >= cutoff
        st.code(f"cutoff = np.percentile(arr, {thresh_pct})\nhigh_mask = arr >= cutoff", language="python")
        st.plotly_chart(plot_2d_heatmap(high_mask.astype(int), title=f"High Performer Flags (>= {cutoff:.1f})", colorscale="Greens"), use_container_width=True)

    st.markdown("#### 3. Total Aggregations & Group Rankings")
    row_sums = np.sum(arr_active, axis=1)
    col_sums = np.sum(arr_active, axis=0)

    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.bar(
            x=df_active.index,
            y=row_sums,
            labels={"x": "Entity", "y": "Total Aggregate"},
            title="Entity Totals (np.sum(axis=1))"
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        fig_c = px.bar(
            x=df_active.columns,
            y=col_sums,
            labels={"x": "Category", "y": "Total Volume"},
            title="Category Totals (np.sum(axis=0))",
            color_discrete_sequence=["#10B981"]
        )
        st.plotly_chart(fig_c, use_container_width=True)


# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown(
    "<center><small style='color: #94A3B8;'>NumPy Data Explorer — Built with Python, NumPy, Pandas, Plotly & Streamlit. Ready for Render deployment.</small></center>",
    unsafe_allow_html=True
)
