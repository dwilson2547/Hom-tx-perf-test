"""Correctness tests comparing Python and Rust htransform implementations."""

from __future__ import annotations

import numpy as np
import pytest

import htransform
import htransform_rs

from tests.fixtures import random_rotation, random_transform

ATOL = 1e-9
BATCH_SIZES = [1, 10, 100, 1000]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_T() -> np.ndarray:
    """Return a single (4, 4) transform."""
    return random_transform(1)[0]


def _rand_batch(n: int) -> np.ndarray:
    """Return (n, 4, 4) batch of transforms."""
    return random_transform(n)


# ---------------------------------------------------------------------------
# compose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_batch_compose_agreement(n: int) -> None:
    """batch_compose: Python and Rust agree."""
    T1s = _rand_batch(n)
    T2s = _rand_batch(n)
    py = htransform.batch_compose(T1s, T2s)
    rs = htransform_rs.batch_compose(T1s, T2s)
    np.testing.assert_allclose(py, rs, atol=ATOL)


def test_single_compose_agreement() -> None:
    """compose: Python and Rust agree."""
    T1 = _rand_T()
    T2 = _rand_T()
    py = htransform.compose(T1, T2)
    rs = htransform_rs.compose(T1, T2)
    np.testing.assert_allclose(py, rs, atol=ATOL)


def test_compose_associativity() -> None:
    """(A @ B) @ C == A @ (B @ C) within float tolerance."""
    A, B, C = _rand_T(), _rand_T(), _rand_T()
    lhs = htransform.compose(htransform.compose(A, B), C)
    rhs = htransform.compose(A, htransform.compose(B, C))
    np.testing.assert_allclose(lhs, rhs, atol=ATOL)


# ---------------------------------------------------------------------------
# invert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_batch_invert_agreement(n: int) -> None:
    """batch_invert: Python and Rust agree."""
    Ts = _rand_batch(n)
    py = htransform.batch_invert(Ts)
    rs = htransform_rs.batch_invert(Ts)
    np.testing.assert_allclose(py, rs, atol=ATOL)


def test_single_invert_agreement() -> None:
    """invert: Python and Rust agree."""
    T = _rand_T()
    py = htransform.invert(T)
    rs = htransform_rs.invert(T)
    np.testing.assert_allclose(py, rs, atol=ATOL)


def test_inversion_identity_python() -> None:
    """compose(T, invert(T)) == I (Python)."""
    T = _rand_T()
    result = htransform.compose(T, htransform.invert(T))
    np.testing.assert_allclose(result, np.eye(4), atol=ATOL)


def test_inversion_identity_rust() -> None:
    """compose(T, invert(T)) == I (Rust)."""
    T = _rand_T()
    result = htransform_rs.compose(T, htransform_rs.invert(T))
    np.testing.assert_allclose(result, np.eye(4), atol=ATOL)


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_batch_inversion_identity_python(n: int) -> None:
    """batch_compose(Ts, batch_invert(Ts)) == I for each element (Python)."""
    Ts = _rand_batch(n)
    inv_Ts = htransform.batch_invert(Ts)
    result = htransform.batch_compose(Ts, inv_Ts)
    identity = np.tile(np.eye(4), (n, 1, 1))
    np.testing.assert_allclose(result, identity, atol=ATOL)


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_batch_inversion_identity_rust(n: int) -> None:
    """batch_compose(Ts, batch_invert(Ts)) == I for each element (Rust)."""
    Ts = _rand_batch(n)
    inv_Ts = htransform_rs.batch_invert(Ts)
    result = htransform_rs.batch_compose(Ts, inv_Ts)
    identity = np.tile(np.eye(4), (n, 1, 1))
    np.testing.assert_allclose(result, identity, atol=ATOL)


# ---------------------------------------------------------------------------
# from_rot_trans / to_rot_trans
# ---------------------------------------------------------------------------


def test_from_to_rot_trans_roundtrip_python() -> None:
    """from_rot_trans / to_rot_trans round-trip (Python)."""
    R = random_rotation(1)[0]
    t = np.array([1.5, -0.3, 2.7])
    T = htransform.from_rot_trans(R, t)
    R2, t2 = htransform.to_rot_trans(T)
    np.testing.assert_allclose(R2, R, atol=ATOL)
    np.testing.assert_allclose(t2, t, atol=ATOL)


def test_from_to_rot_trans_roundtrip_rust() -> None:
    """from_rot_trans / to_rot_trans round-trip (Rust)."""
    R = random_rotation(1)[0]
    t = np.array([1.5, -0.3, 2.7])
    T = htransform_rs.from_rot_trans(R, t)
    R2, t2 = htransform_rs.to_rot_trans(T)
    np.testing.assert_allclose(R2, R, atol=ATOL)
    np.testing.assert_allclose(t2, t, atol=ATOL)


def test_from_rot_trans_agreement() -> None:
    """from_rot_trans: Python and Rust agree."""
    R = random_rotation(1)[0]
    t = np.random.default_rng(7).standard_normal(3)
    py = htransform.from_rot_trans(R, t)
    rs = htransform_rs.from_rot_trans(R, t)
    np.testing.assert_allclose(py, rs, atol=ATOL)


def test_to_rot_trans_agreement() -> None:
    """to_rot_trans: Python and Rust agree."""
    T = _rand_T()
    R_py, t_py = htransform.to_rot_trans(T)
    R_rs, t_rs = htransform_rs.to_rot_trans(T)
    np.testing.assert_allclose(R_py, R_rs, atol=ATOL)
    np.testing.assert_allclose(t_py, t_rs, atol=ATOL)


# ---------------------------------------------------------------------------
# apply_points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_apply_points_agreement(n: int) -> None:
    """apply_points: Python and Rust agree."""
    T = _rand_T()
    pts = np.random.default_rng(99).standard_normal((n, 3))
    py = htransform.apply_points(T, pts)
    rs = htransform_rs.apply_points(T, pts)
    np.testing.assert_allclose(py, rs, atol=ATOL)


def test_apply_points_identity() -> None:
    """Applying identity transform leaves points unchanged."""
    pts = np.random.default_rng(0).standard_normal((50, 3))
    result = htransform.apply_points(np.eye(4), pts)
    np.testing.assert_allclose(result, pts, atol=ATOL)


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------


def test_interpolate_endpoints_python() -> None:
    """interpolate(T1, T2, 0) == T1, interpolate(T1, T2, 1) == T2 (Python)."""
    T1 = _rand_T()
    T2 = _rand_T()
    np.testing.assert_allclose(htransform.interpolate(T1, T2, 0.0), T1, atol=ATOL)
    np.testing.assert_allclose(htransform.interpolate(T1, T2, 1.0), T2, atol=ATOL)


def test_interpolate_endpoints_rust() -> None:
    """interpolate(T1, T2, 0) == T1, interpolate(T1, T2, 1) == T2 (Rust)."""
    T1 = _rand_T()
    T2 = _rand_T()
    np.testing.assert_allclose(htransform_rs.interpolate(T1, T2, 0.0), T1, atol=ATOL)
    np.testing.assert_allclose(htransform_rs.interpolate(T1, T2, 1.0), T2, atol=ATOL)


def test_interpolate_agreement() -> None:
    """interpolate: Python and Rust agree at midpoint."""
    T1 = _rand_T()
    T2 = _rand_T()
    py = htransform.interpolate(T1, T2, 0.5)
    rs = htransform_rs.interpolate(T1, T2, 0.5)
    # SLERP implementations may differ slightly between scipy and nalgebra
    np.testing.assert_allclose(py, rs, atol=1e-6)
