# Lee

A small, readable Lie group library in Python for **SO(2), SO(3), SE(2) and SE(3)**: the groups of 2-D and 3-D rotations and rigid-body poses. It depends only on numpy.

Lee provides everything needed to work with poses on the manifold:

- the hat/vee maps and the exponential and logarithm maps (Exp/Log), with special-case formulas near θ = 0 and θ = π so they stay accurate there;
- left and right Jacobians of SO(3), SE(2) and SE(3), and their inverses;
- adjoints (`Ad`, `ad`);
- composition with `@`, pose inverses, transforming points;
- the ⊞ / ⊟ operators (`boxplus` / `boxminus`);
- building poses from quaternions (SE(3)) or angles (SE(2)).

## What it's for

Lee was written as the maths layer for a **pose graph optimization (PGO) backend for SLAM**, built from scratch while learning the subject. A SLAM backend has to compare poses, compute residuals between them and differentiate those residuals. Doing that correctly on SE(2)/SE(3) is exactly what this library provides. It's used by [Pose-Graph-Optimization-Learning](https://github.com/Vulcan758/Pose-Graph-Optimization-Learning), whose Gauss-Newton solver built on Lee matches GTSAM's results (same steps, same final cost) on standard 2-D and 3-D benchmark datasets.

It's intended for:
- **learning** Lie theory for robotics: the code follows the textbook formulas closely, so it's easy to read alongside notes or a book;
- **prototyping** SLAM backends, pose graph optimization, and other estimation problems on SE(2)/SE(3) in Python;
- **checking** your own derivations: every Exp, Log and Jacobian can be compared against `scipy.linalg.expm` or finite differences, as the examples notebook does.

It is **not** meant for high-performance production use. Each operation is a small numpy computation on Python objects, so a pose-graph edge takes roughly 0.6 ms to linearise. For large problems or real-time systems, use a compiled library such as GTSAM, Sophus or manif.

## Installation

```bash
pip install lee-groups
```

or the latest version straight from GitHub:

```bash
pip install git+https://github.com/Vulcan758/Lee.git
```

To also install what the examples notebook needs (scipy, matplotlib, jupyter):

```bash
pip install "lee-groups[examples]"
```

The package is called `lee-groups` on PyPI, and the module you import is `lee`:

```python
import numpy as np
from lee import SO_n, so_n, se3, SE3, se2, SE2
```

## Conventions

- **Twists are translation first:** ξ = (ρ, φ). For SE(3) that's `[x, y, z, φx, φy, φz]`; for SE(2), `[x, y, θ]`. Some libraries put rotation first, so check before mixing them.
- **Right perturbation:** `X.boxplus(d)` is X · Exp(d), and `Y.boxminus(X)` is Log(X⁻¹ · Y).
- **Quaternions are (x, y, z, w)**, scalar last: the g2o / ROS / scipy order. They're normalised for you.
- **Inputs are numpy arrays** (vectors as 1-D float arrays).
- **Validation:** `SO_n(R)` checks that R is a rotation and raises `ValueError` otherwise. The library skips that check internally where the result is guaranteed valid (`validate=False`).

## Usage

### `SO_n`: rotation matrices

```python
R = SO_n(np.array([[0.0, -1.0, 0.0],
                   [1.0,  0.0, 0.0],
                   [0.0,  0.0, 1.0]]))   # raises ValueError if not a rotation

R.R              # the 3x3 matrix
R.inverse()      # its inverse (the transpose), as a matrix
R.log_map()      # Log: returns an so_n rotation vector
R.log_map().vee  # -> array([0., 0., 1.5708])
```

Works for 2×2 rotations too: `SO_n(R2).log_map().vee` gives `[θ]`.

### `so_n`: rotation vectors (the Lie algebra)

```python
phi = so_n(np.array([0.3, -0.2, 0.9]))   # axis * angle; a 1-vector [θ] for 2-D

phi.vee        # the vector
phi.hat        # the 3x3 skew-symmetric matrix
phi.theta      # rotation angle |phi|
phi.exp_map()  # Exp: returns an SO_n rotation (Rodrigues' formula)

Jl, Jl_inv = phi.compute_left_jacobians()            # J_l and its inverse
Jr, Jr_inv = phi.compute_left_jacobians(right=True)  # J_r and its inverse
```

### `se3`: 3-D twists

```python
xi = se3(np.array([1.0, -0.5, 0.2, 0.3, -0.2, 0.9]))   # (rho, phi)

xi.vee, xi.rho, xi.phi   # the 6-vector, its translation part, its rotation part (so_n)
xi.hat                   # 4x4 matrix
xi.ad()                  # 6x6 adjoint of the algebra
T = xi.exp_map()         # Exp: returns an SE3 pose

Jl, Jl_inv = xi.compute_left_jacobians()
Jr, Jr_inv = xi.compute_left_jacobians(right=True)
```

### `SE3`: 3-D poses

```python
T = SE3(SO_n(R_matrix), np.array([1.0, 2.0, 3.0]))           # from a rotation
T = SE3(np.array([0.0, 0.0, 0.383, 0.924]), np.zeros(3))     # from a quaternion (x, y, z, w)

T.X, T.R, T.t_flat       # 4x4 matrix, 3x3 rotation, translation (T.t is the 3x1 column)

T @ U                    # compose two poses -> SE3
T @ p                    # transform a point (3,) or a cloud (3, N)
T.inverse_pose()         # inverse as an SE3
T.inverse()              # inverse as a 4x4 matrix

T.Log()                  # Log: 6-vector twist (T.log_map() returns the se3 object)
T.boxplus(d)             # T * Exp(d)
U.boxminus(T)            # Log(T^-1 * U)
T.Ad(), T.Ad_inv()       # 6x6 adjoint and its inverse
```

### `se2` / `SE2`: the 2-D versions

Same interface, with 3-vector twists `(x, y, θ)` and 3×3 matrices.

```python
xi = se2(np.array([1.0, -0.5, 0.8]))
T = xi.exp_map()                          # SE2
Jr, Jr_inv = xi.compute_left_jacobians(right=True)

T = SE2(0.8, np.array([1.0, 2.0]))        # from an angle (or an SO_n 2x2 rotation)
T.theta                                   # heading angle
T @ U, T @ np.array([1.0, 0.0])           # compose, transform a point
T.Log(), T.boxplus(d), U.boxminus(T), T.Ad()
```

### Example: a pose-graph residual and its Jacobians

The core of a PGO backend. For an edge measuring pose j relative to pose i:

```python
def between_factor(Xi, Xj, Z):
    T = Xi.inverse_pose() @ Xj
    e = (Z.inverse_pose() @ T).Log()                         # residual
    _, Jr_inv = se3(e).compute_left_jacobians(right=True)     # use se2 for SE2 poses
    A = -Jr_inv @ T.Ad_inv()                                  # d e / d(perturbation of Xi)
    B = Jr_inv                                                # d e / d(perturbation of Xj)
    return e, A, B
```

## Examples notebook

[`examples/examples.ipynb`](examples/examples.ipynb) walks through every class and **checks each feature**:
- Exp/Log against `scipy.linalg.expm`, including near θ = 0 and π;
- the Jacobians through their defining first-order identities;
- composition, inverses and points against 4×4 matrix products;
- the adjoint identity, and the ⊞/⊟ round trips;
- the pose-graph Jacobians above against finite differences, for SE(3) and SE(2);
- a small plot of a 2-D trajectory built with `boxplus`.

## Limitations

- **Exp/Log and the Jacobians are implemented for SO(2), SO(3), SE(2) and SE(3).** `to_hat`/`to_vee` and `SO_n` accept higher dimensions, but the closed-form maps are only valid in 2-D and 3-D.
- **Speed:** pure Python and numpy, one pose at a time (see "What it's for").

## License

MIT. See [LICENSE](LICENSE).
