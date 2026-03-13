# htransform — Polyglot Homogeneous Transform Library

A polyglot library implementing homogeneous transform operations, with a shared
benchmark and correctness test suite that compares performance across language
implementations. The initial scope covers **Python** (NumPy/SciPy) and **Rust**
(via PyO3/nalgebra), with the architecture designed to accommodate additional
languages later.

---

## Repository Structure

```
htransform/
├── python/
│   ├── htransform/
│   │   ├── __init__.py
│   │   └── transforms.py
│   └── pyproject.toml
├── rust/
│   ├── src/
│   │   ├── lib.rs
│   │   └── transforms.rs
│   └── Cargo.toml
├── bench/
│   ├── bench_python.py
│   ├── bench_rust.py
│   └── report.py
├── tests/
│   ├── fixtures.py
│   ├── test_correctness.py
│   └── test_perf.py
└── README.md
```

---

## Operations Contract

Every language implementation implements the following operations:

| Operation | Description |
|---|---|
| `compose(T1, T2)` | Matrix multiply two 4×4 homogeneous transforms |
| `invert(T)` | Invert a homogeneous transform, exploiting rigid-body structure |
| `batch_compose(T1s, T2s)` | Batched compose over arrays of shape `(N, 4, 4)` |
| `batch_invert(Ts)` | Batched invert over `(N, 4, 4)` |
| `apply_points(T, pts)` | Apply transform to a batch of 3-D points, shape `(N, 3)` |
| `from_rot_trans(R, t)` | Construct 4×4 T from rotation matrix `(3, 3)` and translation `(3,)` |
| `to_rot_trans(T)` | Decompose 4×4 T into `(R, t)` |
| `interpolate(T1, T2, alpha)` | SLERP/lerp interpolation between two transforms |

### Inversion identity

For rigid-body transforms:

```
[R | t]^-1 = [R^T | -R^T * t]
[0 | 1]      [0   |  1      ]
```

---

## Setup

### Python

Requirements: Python ≥ 3.10, `numpy`, `scipy`, `pytest`, `pytest-benchmark`.

```bash
pip install numpy scipy pytest pytest-benchmark
pip install -e python/
```

### Rust extension

Requirements: Rust toolchain, `maturin`.

```bash
pip install maturin
# Build and install the wheel:
cd rust/
maturin build --release
pip install target/wheels/htransform_rs-*.whl
```

> **Development workflow:** if you have a virtualenv active you can use
> `maturin develop --release` inside `rust/` for an in-place editable install.

---

## Running Correctness Tests

```bash
pytest tests/test_correctness.py -v
```

All 33 tests should pass.  They verify that:

- Python and Rust outputs agree within `atol=1e-9` (f64).
- `compose(T, invert(T)) == I` for randomly generated rigid-body transforms.
- `from_rot_trans` / `to_rot_trans` round-trip correctly.
- `batch_compose` / `batch_invert` produce per-element identity products.
- `apply_points` with the identity transform leaves points unchanged.
- `interpolate` returns the endpoints at `alpha=0` and `alpha=1`.

---

## Running Benchmarks

### Generate benchmark JSON files

```bash
pytest bench/bench_python.py --benchmark-json=bench/results_python.json
pytest bench/bench_rust.py   --benchmark-json=bench/results_rust.json
```

### Generate the comparison report

```bash
python bench/report.py                          # prints to stdout
python bench/report.py --output bench/report.md  # also writes Markdown
```

### Run performance tests (all batch sizes, both implementations)

```bash
pytest tests/test_perf.py --benchmark-enable
```

---

## Benchmark Report Format

| operation | batch_size | python_mean_ms | rust_mean_ms | speedup |
|-----------|------------|---------------|-------------|---------|
| compose | 1 | — | — | — |
| batch_compose | 1 | — | — | — |
| batch_compose | 100 | — | — | — |
| batch_compose | 1000 | — | — | — |
| batch_compose | 10000 | — | — | — |
| batch_compose | 100000 | — | — | — |
| batch_invert | 1 | — | — | — |
| … | … | … | … | … |

*(Fill in after first benchmark run with `python bench/report.py`.)*

---

## What's Not Included Yet

- f32 variants (architecture supports them; not implemented).
- Lua, C/C++ implementations.
- GPU/CUDA paths.
- PyPI packaging.
- CI pipeline.
