"""Performance benchmarks using pytest-benchmark.

Run with::

    pytest tests/test_perf.py --benchmark-enable

"""

from __future__ import annotations

import numpy as np
import pytest

import htransform
import htransform_rs
from tests.fixtures import random_transform

BATCH_SIZES = [1, 100, 1_000, 10_000, 100_000]


def _impl(name: str):
    if name == "python":
        return htransform
    return htransform_rs


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="compose")
@pytest.mark.parametrize("impl_name", ["python", "rust"])
def test_bench_compose(benchmark, impl_name: str) -> None:
    mod = _impl(impl_name)
    T1 = random_transform(1)[0]
    T2 = random_transform(1)[0]
    benchmark(mod.compose, T1, T2)


# ---------------------------------------------------------------------------
# batch_compose
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="batch_compose")
@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("impl_name", ["python", "rust"])
def test_bench_batch_compose(benchmark, n: int, impl_name: str) -> None:
    mod = _impl(impl_name)
    T1s = random_transform(n)
    T2s = random_transform(n)
    benchmark(mod.batch_compose, T1s, T2s)


# ---------------------------------------------------------------------------
# invert
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="invert")
@pytest.mark.parametrize("impl_name", ["python", "rust"])
def test_bench_invert(benchmark, impl_name: str) -> None:
    mod = _impl(impl_name)
    T = random_transform(1)[0]
    benchmark(mod.invert, T)


# ---------------------------------------------------------------------------
# batch_invert
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="batch_invert")
@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("impl_name", ["python", "rust"])
def test_bench_batch_invert(benchmark, n: int, impl_name: str) -> None:
    mod = _impl(impl_name)
    Ts = random_transform(n)
    benchmark(mod.batch_invert, Ts)


# ---------------------------------------------------------------------------
# apply_points
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="apply_points")
@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("impl_name", ["python", "rust"])
def test_bench_apply_points(benchmark, n: int, impl_name: str) -> None:
    mod = _impl(impl_name)
    T = random_transform(1)[0]
    pts = np.random.default_rng(0).standard_normal((n, 3))
    benchmark(mod.apply_points, T, pts)


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="interpolate")
@pytest.mark.parametrize("impl_name", ["python", "rust"])
def test_bench_interpolate(benchmark, impl_name: str) -> None:
    mod = _impl(impl_name)
    T1 = random_transform(1)[0]
    T2 = random_transform(1)[0]
    benchmark(mod.interpolate, T1, T2, 0.5)
