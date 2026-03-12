//! Homogeneous transform operations — core implementation.
//!
//! All transforms are rigid-body 4×4 matrices with the bottom row fixed to
//! `[0, 0, 0, 1]`.  We exploit this structure for fast inversion.

use nalgebra::{Matrix3, Matrix4, Quaternion, UnitQuaternion, Vector3};
use ndarray::{s, Array3, ArrayView3};
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Helpers: convert flat row-major slice ↔ nalgebra types
// ---------------------------------------------------------------------------

/// Build a `Matrix4<f64>` from a 16-element row-major slice.
#[inline]
pub fn slice_to_mat4(s: &[f64]) -> Matrix4<f64> {
    // nalgebra Matrix4::new takes arguments in row-major order.
    Matrix4::new(
        s[0], s[1], s[2], s[3],
        s[4], s[5], s[6], s[7],
        s[8], s[9], s[10], s[11],
        s[12], s[13], s[14], s[15],
    )
}

/// Write a `Matrix4<f64>` into a 16-element row-major slice.
#[inline]
pub fn mat4_to_slice(m: &Matrix4<f64>, out: &mut [f64]) {
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = m[(r, c)];
        }
    }
}

// ---------------------------------------------------------------------------
// Single-transform operations
// ---------------------------------------------------------------------------

/// Multiply two 4×4 homogeneous transforms.
pub fn compose_mat(t1: &Matrix4<f64>, t2: &Matrix4<f64>) -> Matrix4<f64> {
    t1 * t2
}

/// Invert a rigid-body transform using R^T structure.
///
/// ```text
/// [R | t]^-1 = [R^T | -R^T t]
/// [0 | 1]      [0   |  1     ]
/// ```
pub fn invert_mat(t: &Matrix4<f64>) -> Matrix4<f64> {
    let r = t.fixed_view::<3, 3>(0, 0);
    let tv = t.fixed_view::<3, 1>(0, 3);
    let rt = r.transpose();
    let neg_rt_t = -(rt * tv);
    let mut result = Matrix4::<f64>::identity();
    result.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
    result.fixed_view_mut::<3, 1>(0, 3).copy_from(&neg_rt_t);
    result
}

/// Construct a 4×4 transform from a 3×3 rotation and translation vector.
pub fn from_rot_trans_mat(r: &Matrix3<f64>, t: &Vector3<f64>) -> Matrix4<f64> {
    let mut result = Matrix4::<f64>::identity();
    result.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    result.fixed_view_mut::<3, 1>(0, 3).copy_from(t);
    result
}

/// Apply a 4×4 transform to a batch of 3-D points stored as (N, 3) in row-major order.
pub fn apply_points_impl(t: &Matrix4<f64>, pts: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let r = t.fixed_view::<3, 3>(0, 0);
    let tv = t.fixed_view::<3, 1>(0, 3);
    pts.iter()
        .map(|p| {
            let v = Vector3::new(p[0], p[1], p[2]);
            let out = r * v + tv;
            [out[0], out[1], out[2]]
        })
        .collect()
}

/// Interpolate between two transforms using SLERP (rotation) + lerp (translation).
pub fn interpolate_mat(t1: &Matrix4<f64>, t2: &Matrix4<f64>, alpha: f64) -> Matrix4<f64> {
    let r1 = t1.fixed_view::<3, 3>(0, 0);
    let r2 = t2.fixed_view::<3, 3>(0, 0);
    let tv1 = t1.fixed_view::<3, 1>(0, 3);
    let tv2 = t2.fixed_view::<3, 1>(0, 3);

    let q1 = UnitQuaternion::from_matrix(&r1.into());
    let q2 = UnitQuaternion::from_matrix(&r2.into());
    let q_interp = q1.slerp(&q2, alpha);
    let r_interp = q_interp.to_rotation_matrix();

    let t_interp: Vector3<f64> = tv1 + alpha * (tv2 - tv1);
    from_rot_trans_mat(r_interp.matrix(), &t_interp)
}

// ---------------------------------------------------------------------------
// Batch operations over (N, 4, 4) ndarray::Array3
// ---------------------------------------------------------------------------

/// Batch compose: element-wise T1[i] @ T2[i].
pub fn batch_compose_impl(t1s: &ArrayView3<f64>, t2s: &ArrayView3<f64>) -> Array3<f64> {
    let n = t1s.shape()[0];
    // Compute results in parallel, collect into a flat Vec.
    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let a: Vec<f64> = t1s.slice(s![i, .., ..]).iter().cloned().collect();
            let b: Vec<f64> = t2s.slice(s![i, .., ..]).iter().cloned().collect();
            let ma = slice_to_mat4(&a);
            let mb = slice_to_mat4(&b);
            let mc = ma * mb;
            let mut row = vec![0f64; 16];
            mat4_to_slice(&mc, &mut row);
            row
        })
        .collect();
    Array3::from_shape_vec((n, 4, 4), flat).unwrap()
}

/// Batch invert: element-wise invert T[i].
pub fn batch_invert_impl(ts: &ArrayView3<f64>) -> Array3<f64> {
    let n = ts.shape()[0];
    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let a: Vec<f64> = ts.slice(s![i, .., ..]).iter().cloned().collect();
            let ma = slice_to_mat4(&a);
            let mi = invert_mat(&ma);
            let mut row = vec![0f64; 16];
            mat4_to_slice(&mi, &mut row);
            row
        })
        .collect();
    Array3::from_shape_vec((n, 4, 4), flat).unwrap()
}

// ---------------------------------------------------------------------------
// Quaternion helpers (exposed for testing)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn unit_quat_from_wxyz(wxyz: [f64; 4]) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]))
}
