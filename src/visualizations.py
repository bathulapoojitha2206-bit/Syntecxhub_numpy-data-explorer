"""Module for interactive Plotly and Matplotlib data visualizations.

Provides rich visual representations for 1D vectors, 2D heatmaps/matrices,
3D surface/scatter representations, slicing mask highlights, statistical
distribution plots, and performance benchmarking charts.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Numerical computing palette tokens
COLOR_PRIMARY = "#3B82F6"      # Indigo / Electric Blue
COLOR_ACCENT = "#10B981"       # Emerald Teal
COLOR_WARN = "#F59E0B"         # Amber
COLOR_DANGER = "#EF4444"       # Coral Red
COLOR_PURPLE = "#8B5CF6"       # Violet
COLOR_BG_DARK = "#0F172A"      # Slate 900
COLOR_GRID = "#E2E8F0"


def plot_1d_array(
    arr: np.ndarray,
    title: str = "1D Array Visualization",
    chart_type: str = "bar"
) -> go.Figure:
    """Plot 1D array as an interactive bar, line, or stem plot.

    Args:
        arr: 1D NumPy array.
        title: Chart title.
        chart_type: 'bar', 'line', 'scatter'.

    Returns:
        go.Figure: Plotly figure.
    """
    flat = arr.ravel()
    indices = [f"[{i}]" for i in range(len(flat))]

    if chart_type == "bar":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=indices,
                    y=flat,
                    text=np.round(flat, 2),
                    textposition="auto",
                    marker=dict(
                        color=flat,
                        colorscale="Blues",
                        showscale=True,
                        colorbar=dict(title="Value")
                    ),
                    hovertemplate="<b>Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>"
                )
            ]
        )
    elif chart_type == "line":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(flat))),
                    y=flat,
                    mode="lines+markers",
                    line=dict(color=COLOR_PRIMARY, width=3),
                    marker=dict(size=8, color=COLOR_ACCENT),
                    hovertemplate="<b>Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>"
                )
            ]
        )
    else:  # scatter
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(flat))),
                    y=flat,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=flat,
                        colorscale="Viridis",
                        showscale=True
                    ),
                    hovertemplate="<b>Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>"
                )
            ]
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1E293B")),
        xaxis=dict(title="Array Index", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="Numerical Value", showgrid=True, gridcolor=COLOR_GRID),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_2d_heatmap(
    arr: np.ndarray,
    title: str = "2D Matrix Heatmap",
    colorscale: str = "Viridis",
    show_values: bool = True
) -> go.Figure:
    """Render 2D matrix as an annotated heatmap with hover inspection.

    Args:
        arr: 2D NumPy array.
        title: Title of figure.
        colorscale: Plotly colorscale (e.g. 'Viridis', 'Plasma', 'Blues', 'RdBu').
        show_values: Whether to print numbers in matrix cells.

    Returns:
        go.Figure: Plotly figure.
    """
    if arr.ndim != 2:
        raise ValueError(f"Heatmap requires 2D array, received {arr.ndim}D.")

    n_rows, n_cols = arr.shape
    x_labels = [f"Col {j}" for j in range(n_cols)]
    y_labels = [f"Row {i}" for i in range(n_rows)]

    annotations = []
    if show_values and n_rows <= 25 and n_cols <= 25:
        # Determine contrast text color based on cell intensity
        min_v = np.nanmin(arr)
        max_v = np.nanmax(arr)
        mid_v = (min_v + max_v) / 2.0 if max_v != min_v else min_v

        for i in range(n_rows):
            for j in range(n_cols):
                val = arr[i, j]
                val_str = f"{val:.2f}" if isinstance(val, (float, np.floating)) else str(val)
                text_col = "white" if (not np.isnan(val) and val < mid_v) else "black"
                annotations.append(
                    dict(
                        x=x_labels[j],
                        y=y_labels[i],
                        text=val_str,
                        font=dict(color=text_col, size=11),
                        showarrow=False
                    )
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            colorbar=dict(title="Value"),
            hovertemplate="<b>%{y}, %{x}</b><br>Value: %{z:.4f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text=f"{title} (Shape: {n_rows} × {n_cols})", font=dict(size=16)),
        annotations=annotations,
        yaxis=dict(autorange="reversed"),  # Row 0 on top
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=max(360, min(n_rows * 40 + 100, 650))
    )
    return fig


def plot_sliced_submatrix(
    arr: np.ndarray,
    r_slice: slice,
    c_slice: slice,
    title: str = "Active Slicing Mask"
) -> go.Figure:
    """Highlight active slice elements inside the full matrix with distinct coloring.

    Args:
        arr: 2D array.
        r_slice: Row slice.
        c_slice: Column slice.
        title: Figure title.

    Returns:
        go.Figure: Plotly figure showing active vs inactive elements.
    """
    n_rows, n_cols = arr.shape
    # Create mask: 1 for sliced elements, 0 for excluded
    mask = np.zeros((n_rows, n_cols), dtype=int)
    mask[r_slice, c_slice] = 1

    x_labels = [f"Col {j}" for j in range(n_cols)]
    y_labels = [f"Row {i}" for i in range(n_rows)]

    annotations = []
    customdata = []
    for i in range(n_rows):
        row_custom = []
        for j in range(n_cols):
            val = arr[i, j]
            is_active = (mask[i, j] == 1)
            status_text = "SELECTED IN SLICE" if is_active else "Excluded"
            row_custom.append(f"{val} ({status_text})")
            
            font_col = "white" if is_active else "#94A3B8"
            font_weight = "bold" if is_active else "normal"
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=f"<b>{val:.2f}</b>" if is_active else f"{val:.2f}",
                    font=dict(color=font_col, size=12),
                    showarrow=False
                )
            )
        customdata.append(row_custom)

    fig = go.Figure(
        data=go.Heatmap(
            z=mask,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#E2E8F0"], [1, COLOR_PRIMARY]],
            showscale=False,
            customdata=customdata,
            hovertemplate="<b>%{y}, %{x}</b><br>%{customdata}<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        annotations=annotations,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=max(360, min(n_rows * 40 + 100, 650))
    )
    return fig


def plot_distribution_analysis(
    arr: np.ndarray,
    title: str = "Data Distribution & Summary",
    stats_dict: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """Plot interactive histogram with KDE and mean/median indicators.

    Args:
        arr: Array of numbers.
        title: Figure title.
        stats_dict: Precomputed statistics dict (mean, median, etc.).

    Returns:
        go.Figure: Plotly figure.
    """
    flat = arr.ravel().astype(float)
    flat = flat[~np.isnan(flat)]  # filter NaNs safely

    fig = go.Figure()
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=flat,
            name="Element Frequency",
            nbinsx=25,
            marker=dict(color="rgba(59, 130, 246, 0.7)", line=dict(color=COLOR_PRIMARY, width=1.5)),
            hovertemplate="<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>"
        )
    )

    mean_val = float(np.mean(flat))
    median_val = float(np.median(flat))

    # Add vertical lines for Mean and Median
    fig.add_vline(
        x=mean_val,
        line_width=3,
        line_dash="dash",
        line_color=COLOR_DANGER,
        annotation_text=f"Mean: {mean_val:.2f}",
        annotation_position="top left"
    )
    fig.add_vline(
        x=median_val,
        line_width=3,
        line_dash="dot",
        line_color=COLOR_ACCENT,
        annotation_text=f"Median: {median_val:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Value", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="Frequency Count", showgrid=True, gridcolor=COLOR_GRID),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        bargap=0.05
    )
    return fig


def plot_benchmark_comparison(
    bench_result: Dict[str, Any]
) -> go.Figure:
    """Render side-by-side execution time comparison bar chart with speedup annotation.

    Args:
        bench_result: Dict containing list_time_ms, numpy_time_ms, speedup_ratio, operation.

    Returns:
        go.Figure: Plotly figure.
    """
    labels = ["Python Standard List", "NumPy Vectorized Array"]
    times = [bench_result["list_time_ms"], bench_result["numpy_time_ms"]]
    colors = ["#EF4444", "#10B981"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=times,
                text=[f"{t:.3f} ms" for t in times],
                textposition="auto",
                marker_color=colors,
                hovertemplate="<b>%{x}:</b> %{y:.4f} ms<extra></extra>"
            )
        ]
    )

    speedup = bench_result["speedup_ratio"]
    fig.update_layout(
        title=dict(
            text=f"Execution Time Comparison ({bench_result['operation'].replace('_', ' ').title()}) — Size: {bench_result['size']:,}",
            font=dict(size=15)
        ),
        yaxis=dict(title="Median Time (milliseconds)", showgrid=True, gridcolor=COLOR_GRID),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=360,
        annotations=[
            dict(
                x=labels[1],
                y=times[1],
                text=f"🚀 <b>{speedup:.1f}x Faster!</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLOR_ACCENT,
                ax=0,
                ay=-45,
                font=dict(size=13, color=COLOR_ACCENT)
            )
        ] if speedup > 1.2 else []
    )
    return fig


def plot_scaling_curves(scaling_data: List[Dict[str, Any]]) -> go.Figure:
    """Plot multi-size scaling line chart comparing Python List vs NumPy across orders of magnitude.

    Args:
        scaling_data: List of benchmark dicts across sizes.

    Returns:
        go.Figure: Plotly figure.
    """
    sizes = [d["size"] for d in scaling_data]
    py_times = [d["list_time_ms"] for d in scaling_data]
    np_times = [d["numpy_time_ms"] for d in scaling_data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=py_times,
            mode="lines+markers",
            name="Python List",
            line=dict(color=COLOR_DANGER, width=3),
            marker=dict(size=9)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=np_times,
            mode="lines+markers",
            name="NumPy Array",
            line=dict(color=COLOR_ACCENT, width=3),
            marker=dict(size=9)
        )
    )

    fig.update_layout(
        title=dict(text="Scaling Trajectory: Execution Time vs Input Size (N)", font=dict(size=16)),
        xaxis=dict(title="Element Count (N)", type="log", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="Execution Time (ms)", type="log", showgrid=True, gridcolor=COLOR_GRID),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        legend=dict(x=0.05, y=0.95),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380
    )
    return fig


def plot_memory_comparison(mem_info: Dict[str, Any]) -> go.Figure:
    """Visualize memory footprint disparity between Python lists and NumPy arrays.

    Args:
        mem_info: Dict from measure_memory_footprint.

    Returns:
        go.Figure: Plotly figure.
    """
    labels = ["Python List Footprint", "NumPy Array Footprint"]
    bytes_vals = [mem_info["list_bytes"], mem_info["numpy_bytes"]]
    labels_fmt = [mem_info["list_formatted"], mem_info["numpy_formatted"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=bytes_vals,
                text=labels_fmt,
                textposition="auto",
                marker_color=[COLOR_WARN, COLOR_PRIMARY]
            )
        ]
    )

    fig.update_layout(
        title=dict(text=f"RAM Memory Consumption Comparison ({mem_info['size']:,} elements)", font=dict(size=15)),
        yaxis=dict(title="Bytes Allocated in RAM", showgrid=True, gridcolor=COLOR_GRID),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )
    return fig
