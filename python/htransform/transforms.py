"""Homogeneous transform operations implemented with NumPy.

All 4×4 transforms are assumed to be rigid-body (rotation + translation) with
the bottom row fixed to ``[0, 0, 0, 1]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Single-transform operations
# ---------------------------------------------------------------------------


def compose(T1: NDArray[np.float64], T2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two 4×4 homogeneous transforms.

    Parameters
    ----------
    T1:
        First transform, shape ``(4, 4)``.
    T2:
        Second transform, shape ``(4, 4)``.

    Returns
    -------
    NDArray[np.float64]
        ``T1 @ T2``, shape ``(4, 4)``.
    """
    return T1 @ T2


def invert(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a rigid-body homogeneous transform.

    Exploits the identity::

        [R | t]^-1 = [R^T | -R^T t]
        [0 | 1]      [0   |  1     ]

    which avoids a general LU decomposition.

    Parameters
    ----------
    T:
        Rigid-body transform, shape ``(4, 4)``.

    Returns
    -------
    NDArray[np.float64]
        Inverse transform, shape ``(4, 4)``.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    RT = R.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = RT
    result[:3, 3] = -(RT @ t)
    return result


def from_rot_trans(
    R: NDArray[np.float64], t: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Construct a 4×4 homogeneous transform from rotation and translation.

    Parameters
    ----------
    R:
        Rotation matrix, shape ``(3, 3)``.
    t:
        Translation vector, shape ``(3,)``.

    Returns
    -------
    NDArray[np.float64]
        Homogeneous transform, shape ``(4, 4)``.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def to_rot_trans(
    T: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Decompose a 4×4 homogeneous transform into rotation and translation.

    Parameters
    ----------
    T:
        Homogeneous transform, shape ``(4, 4)``.

    Returns
    -------
    R:
        Rotation matrix, shape ``(3, 3)``.
    t:
        Translation vector, shape ``(3,)``.
    """
    return T[:3, :3].copy(), T[:3, 3].copy()


def apply_points(
    T: NDArray[np.float64], pts: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a homogeneous transform to a batch of 3-D points.

    Parameters
    ----------
    T:
        Rigid-body transform, shape ``(4, 4)``.
    pts:
        Points to transform, shape ``(N, 3)``.

    Returns
    -------
    NDArray[np.float64]
        Transformed points, shape ``(N, 3)``.
    """
    # Pad with homogeneous coordinate 1, apply, strip.
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N, 4)
    return (pts_h @ T.T)[:, :3]


def interpolate(
    T1: NDArray[np.float64], T2: NDArray[np.float64], alpha: float
) -> NDArray[np.float64]:
    """Interpolate between two rigid-body transforms using SLERP + lerp.

    Rotation is interpolated via SLERP; translation is interpolated linearly.

    Parameters
    ----------
    T1:
        Start transform, shape ``(4, 4)``.
    T2:
        End transform, shape ``(4, 4)``.
    alpha:
        Interpolation parameter in ``[0, 1]``.  ``0`` returns ``T1``,
        ``1`` returns ``T2``.

    Returns
    -------
    NDArray[np.float64]
        Interpolated transform, shape ``(4, 4)``.
    """
    R1, t1 = to_rot_trans(T1)
    R2, t2 = to_rot_trans(T2)

    rotations = Rotation.from_matrix(np.stack([R1, R2]))
    slerp = Slerp([0.0, 1.0], rotations)
    R_interp = slerp(alpha).as_matrix()

    t_interp = t1 + alpha * (t2 - t1)
    return from_rot_trans(R_interp, t_interp)


# ---------------------------------------------------------------------------
# Batch operations  (N, 4, 4)
# ---------------------------------------------------------------------------


def batch_compose(
    T1s: NDArray[np.float64], T2s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Element-wise compose two batches of 4×4 transforms.

    Parameters
    ----------
    T1s:
        First batch of transforms, shape ``(N, 4, 4)``.
    T2s:
        Second batch of transforms, shape ``(N, 4, 4)``.

    Returns
    -------
    NDArray[np.float64]
        Composed transforms, shape ``(N, 4, 4)``.
    """
    return np.einsum("nij,njk->nik", T1s, T2s)


def batch_invert(Ts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Invert a batch of rigid-body transforms.

    Exploits the rigid-body inversion identity::

        [R | t]^-1 = [R^T | -R^T t]
        [0 | 1]      [0   |  1     ]

    Parameters
    ----------
    Ts:
        Batch of transforms, shape ``(N, 4, 4)``.

    Returns
    -------
    NDArray[np.float64]
        Inverted transforms, shape ``(N, 4, 4)``.
    """
    Rs = Ts[:, :3, :3]          # (N, 3, 3)
    ts = Ts[:, :3, 3]           # (N, 3)
    RTs = np.swapaxes(Rs, -1, -2)                    # (N, 3, 3)
    neg_RTt = -np.einsum("nij,nj->ni", RTs, ts)     # (N, 3)

    N = Ts.shape[0]
    result = np.zeros((N, 4, 4), dtype=np.float64)
    result[:, :3, :3] = RTs
    result[:, :3, 3] = neg_RTt
    result[:, 3, 3] = 1.0
    return result
