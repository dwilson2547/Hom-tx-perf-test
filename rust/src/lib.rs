//! PyO3 bindings for the `htransform_rs` extension module.

use ndarray::ArrayView3;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::transforms::{
    apply_points_impl, batch_compose_impl, batch_invert_impl, from_rot_trans_mat, invert_mat,
    interpolate_mat, mat4_to_slice, slice_to_mat4,
};

mod transforms;

// ---------------------------------------------------------------------------
// Single-transform bindings
// ---------------------------------------------------------------------------

/// Multiply two 4×4 homogeneous transforms.
#[pyfunction]
fn compose<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray2<f64>,
    t2: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let a: Vec<f64> = t1.as_array().iter().cloned().collect();
    let b: Vec<f64> = t2.as_array().iter().cloned().collect();
    let ma = slice_to_mat4(&a);
    let mb = slice_to_mat4(&b);
    let mc = ma * mb;
    let mut out = vec![0f64; 16];
    mat4_to_slice(&mc, &mut out);
    let arr = ndarray::Array2::from_shape_vec((4, 4), out).unwrap();
    arr.into_pyarray_bound(py)
}

/// Invert a rigid-body 4×4 transform using R^T structure.
#[pyfunction]
fn invert<'py>(py: Python<'py>, t: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
    let a: Vec<f64> = t.as_array().iter().cloned().collect();
    let ma = slice_to_mat4(&a);
    let mi = invert_mat(&ma);
    let mut out = vec![0f64; 16];
    mat4_to_slice(&mi, &mut out);
    let arr = ndarray::Array2::from_shape_vec((4, 4), out).unwrap();
    arr.into_pyarray_bound(py)
}

/// Construct a 4×4 transform from a (3,3) rotation matrix and (3,) translation.
#[pyfunction]
fn from_rot_trans<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<f64>,
    t: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let ra = r.as_array();
    let ta = t.as_array();
    let rm = nalgebra::Matrix3::new(
        ra[[0, 0]], ra[[0, 1]], ra[[0, 2]],
        ra[[1, 0]], ra[[1, 1]], ra[[1, 2]],
        ra[[2, 0]], ra[[2, 1]], ra[[2, 2]],
    );
    let tv = nalgebra::Vector3::new(ta[0], ta[1], ta[2]);
    let m = from_rot_trans_mat(&rm, &tv);
    let mut out = vec![0f64; 16];
    mat4_to_slice(&m, &mut out);
    let arr = ndarray::Array2::from_shape_vec((4, 4), out).unwrap();
    arr.into_pyarray_bound(py)
}

/// Decompose a 4×4 transform into (R (3,3), t (3,)).
#[pyfunction]
fn to_rot_trans<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<f64>,
) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>) {
    let a = t.as_array();
    let mut r_data = vec![0f64; 9];
    for row in 0..3usize {
        for col in 0..3usize {
            r_data[row * 3 + col] = a[[row, col]];
        }
    }
    let t_data = vec![a[[0, 3]], a[[1, 3]], a[[2, 3]]];
    let r_arr = ndarray::Array2::from_shape_vec((3, 3), r_data).unwrap();
    let t_arr = ndarray::Array1::from_vec(t_data);
    (r_arr.into_pyarray_bound(py), t_arr.into_pyarray_bound(py))
}

/// Apply a 4×4 transform to a batch of 3-D points (N, 3).
#[pyfunction]
fn apply_points<'py>(
    py: Python<'py>,
    t: PyReadonlyArray2<f64>,
    pts: PyReadonlyArray2<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let a: Vec<f64> = t.as_array().iter().cloned().collect();
    let ma = slice_to_mat4(&a);
    let pa = pts.as_array();
    let n = pa.shape()[0];
    let points: Vec<[f64; 3]> = (0..n).map(|i| [pa[[i, 0]], pa[[i, 1]], pa[[i, 2]]]).collect();
    let out_pts = apply_points_impl(&ma, &points);
    let flat: Vec<f64> = out_pts.iter().flat_map(|p| p.iter().cloned()).collect();
    let arr = ndarray::Array2::from_shape_vec((n, 3), flat).unwrap();
    arr.into_pyarray_bound(py)
}

/// Interpolate between two transforms using SLERP + lerp.
#[pyfunction]
fn interpolate<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray2<f64>,
    t2: PyReadonlyArray2<f64>,
    alpha: f64,
) -> Bound<'py, PyArray2<f64>> {
    let a: Vec<f64> = t1.as_array().iter().cloned().collect();
    let b: Vec<f64> = t2.as_array().iter().cloned().collect();
    let ma = slice_to_mat4(&a);
    let mb = slice_to_mat4(&b);
    let mi = interpolate_mat(&ma, &mb, alpha);
    let mut out = vec![0f64; 16];
    mat4_to_slice(&mi, &mut out);
    let arr = ndarray::Array2::from_shape_vec((4, 4), out).unwrap();
    arr.into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// Batch bindings
// ---------------------------------------------------------------------------

/// Batch compose: element-wise T1[i] @ T2[i], shapes (N,4,4).
#[pyfunction]
fn batch_compose<'py>(
    py: Python<'py>,
    t1s: PyReadonlyArray3<f64>,
    t2s: PyReadonlyArray3<f64>,
) -> Bound<'py, PyArray3<f64>> {
    let a = t1s.as_array();
    let b = t2s.as_array();
    let a_owned;
    let b_owned;
    let av: ArrayView3<f64> = if a.is_standard_layout() {
        a
    } else {
        a_owned = a.to_owned();
        a_owned.view()
    };
    let bv: ArrayView3<f64> = if b.is_standard_layout() {
        b
    } else {
        b_owned = b.to_owned();
        b_owned.view()
    };
    let result = batch_compose_impl(&av, &bv);
    result.into_pyarray_bound(py)
}

/// Batch invert: element-wise invert T[i], shape (N,4,4).
#[pyfunction]
fn batch_invert<'py>(
    py: Python<'py>,
    ts: PyReadonlyArray3<f64>,
) -> Bound<'py, PyArray3<f64>> {
    let a = ts.as_array();
    let a_owned;
    let view: ArrayView3<f64> = if a.is_standard_layout() {
        a
    } else {
        a_owned = a.to_owned();
        a_owned.view()
    };
    let result = batch_invert_impl(&view);
    result.into_pyarray_bound(py)
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn htransform_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compose, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(from_rot_trans, m)?)?;
    m.add_function(wrap_pyfunction!(to_rot_trans, m)?)?;
    m.add_function(wrap_pyfunction!(apply_points, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compose, m)?)?;
    m.add_function(wrap_pyfunction!(batch_invert, m)?)?;
    Ok(())
}
