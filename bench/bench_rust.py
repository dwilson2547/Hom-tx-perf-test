"""Benchmark suite for the Rust (htransform_rs) htransform implementation.

Usage::

    pytest bench/bench_rust.py --benchmark-json=bench/results_rust.json

"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import htransform_rs
from tests.fixtures import random_transform

BATCH_SIZES = [1, 10, 100, 1_000, 10_000, 100_000]


@pytest.mark.benchmark(group="compose")
def test_compose(benchmark) -> None:
    T1 = random_transform(1)[0]
    T2 = random_transform(1)[0]
    benchmark(htransform_rs.compose, T1, T2)


@pytest.mark.benchmark(group="invert")
def test_invert(benchmark) -> None:
    T = random_transform(1)[0]
    benchmark(htransform_rs.invert, T)


@pytest.mark.benchmark(group="batch_compose")
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_batch_compose(benchmark, n: int) -> None:
    T1s = random_transform(n)
    T2s = random_transform(n)
    benchmark(htransform_rs.batch_compose, T1s, T2s)


@pytest.mark.benchmark(group="batch_invert")
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_batch_invert(benchmark, n: int) -> None:
    Ts = random_transform(n)
    benchmark(htransform_rs.batch_invert, Ts)


@pytest.mark.benchmark(group="apply_points")
@pytest.mark.parametrize("n", BATCH_SIZES)
def test_apply_points(benchmark, n: int) -> None:
    T = random_transform(1)[0]
    pts = np.random.default_rng(0).standard_normal((n, 3))
    benchmark(htransform_rs.apply_points, T, pts)


@pytest.mark.benchmark(group="interpolate")
def test_interpolate(benchmark) -> None:
    T1 = random_transform(1)[0]
    T2 = random_transform(1)[0]
    benchmark(htransform_rs.interpolate, T1, T2, 0.5)
