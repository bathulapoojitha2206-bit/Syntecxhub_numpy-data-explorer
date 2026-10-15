"""Unit tests for performance benchmarking and memory measurement."""

import pytest
from src.performance_benchmark import (
    measure_memory_footprint,
    benchmark_operation,
    run_scaling_benchmark,
)


def test_measure_memory_footprint():
    mem = measure_memory_footprint(size=1000)
    assert mem["size"] == 1000
    assert mem["list_bytes"] > mem["numpy_bytes"]
    assert mem["saving_ratio"] > 1.0
    assert "KB" in mem["list_formatted"] or "B" in mem["list_formatted"]


@pytest.mark.parametrize("op", ["creation", "addition", "multiplication", "sum", "scalar_mult"])
def test_benchmark_operations(op):
    res = benchmark_operation(op, size=500, warmups=1, repeats=2)
    assert res["operation"] == op
    assert res["list_time_ms"] >= 0.0
    assert res["numpy_time_ms"] >= 0.0
    assert res["speedup_ratio"] > 0.0
    assert res["verification_passed"] is True


def test_benchmark_unknown_op():
    with pytest.raises(ValueError, match="Unknown benchmark operation"):
        benchmark_operation("invalid_op", size=100)


def test_run_scaling_benchmark():
    scaling = run_scaling_benchmark("addition", sizes=[100, 500])
    assert len(scaling) == 2
    assert scaling[0]["size"] == 100
    assert scaling[1]["size"] == 500
    assert scaling[0]["verification_passed"] is True
