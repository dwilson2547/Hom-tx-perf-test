"""Deterministic test data for correctness and performance tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

RNG = np.random.default_rng(seed=42)


def random_rotation(n: int = 1) -> NDArray[np.float64]:
    """Generate valid SO(3) rotation matrices via QR decomposition.

    Parameters
    ----------
    n:
        Number of rotation matrices to generate.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(n, 3, 3)``.  For a single matrix use ``result[0]``.
    """
    Rs = []
    for _ in range(n):
        A = RNG.standard_normal((3, 3))
        Q, _ = np.linalg.qr(A)
        # Ensure det = +1 (proper rotation).
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        Rs.append(Q)
    return np.stack(Rs, axis=0).astype(np.float64)


def random_transform(n: int = 1) -> NDArray[np.float64]:
    """Generate valid 4×4 homogeneous transforms.

    Parameters
    ----------
    n:
        Number of transforms to generate.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(n, 4, 4)``.  For a single matrix use ``result[0]``.
    """
    Rs = random_rotation(n)                                 # (n, 3, 3)
    ts = RNG.standard_normal((n, 3)).astype(np.float64)    # (n, 3)

    Ts = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))   # (n, 4, 4)
    Ts[:, :3, :3] = Rs
    Ts[:, :3, 3] = ts
    return Ts
