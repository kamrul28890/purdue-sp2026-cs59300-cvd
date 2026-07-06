# CS59300-CVD Assignment 4

**Topics:** Fundamental Matrix Estimation · Camera Calibration · Triangulation · Affine Factorization (Tomasi-Kanade)

---

## Requirements

Python 3.10+ and the following packages:

| Package      | Version tested |
|-------------|----------------|
| torch        | 2.11.0         |
| torchvision  | 0.22.0         |
| matplotlib   | 3.x            |
| numpy        | 1.x / 2.x      |

Install everything with:

```bash
pip install torch torchvision matplotlib numpy
```

---

## Repository Structure

```
hw4/
├── README.md
├── part1/                        # Fundamental matrix, calibration, triangulation
│   ├── main.py                   # Entry point — unchanged from starter code
│   ├── camera.py                 # ★ Implementation: fit_fundamental_*, camera_calibration, triangulation
│   ├── utils/
│   │   ├── io_helper.py
│   │   ├── math_helper.py
│   │   └── figure_helper.py
│   ├── data/
│   │   ├── lab1.jpg, lab2.jpg
│   │   ├── lab_matches.txt
│   │   ├── lab_3d.txt
│   │   ├── library1.jpg, library2.jpg
│   │   ├── library_matches.txt
│   │   ├── library1_camera.txt
│   │   └── library2_camera.txt
│   └── results/                  # Auto-created; output images written here
│
└── part2/                        # Affine factorization (Tomasi-Kanade)
    ├── main.py                   # Entry point — unchanged from starter code
    ├── affine.py                 # ★ Implementation: normalize_measurements, get_structure_and_motion, get_Q
    ├── utils/
    │   ├── io_helper.py
    │   └── figure_helper.py
    ├── data/
    │   ├── measurement_matrix.txt
    │   ├── frame*.jpg            # 101 frames of the hotel sequence
    │   └── readme.txt
    └── results/                  # Auto-created; output images written here
```

---

## How to Run

### Part 1 — Fundamental Matrix, Camera Calibration, Triangulation

Run from the `part1/` directory:

```bash
cd part1

# Lab image pair (fundamental matrix + calibration + triangulation)
python main.py -i lab

# Library image pair (fundamental matrix + triangulation with given cameras)
python main.py -i library
```

**Outputs in `part1/results/`:**

| File | Description |
|------|-------------|
| `{name}-matches.png` | Side-by-side images with correspondence arrows |
| `{name}-F_unnorm.png` | Epipolar lines on image 2 (unnormalized F) |
| `{name}-F_norm.png` | Epipolar lines on image 2 (normalized F̄) |

**Console output includes:**
- Fundamental matrices F and F̄ (scaled so F₃₃ = 1)
- Unnormalized vs. normalized MSE (epipolar line distance)
- Camera projection matrices P₁, P₂
- Camera calibration 2D MSE (lab only)
- Triangulation 3D MSE (lab only)
- 2D reprojection errors from triangulated 3D points

---

### Part 2 — Affine Factorization

Run from the `part2/` directory:

```bash
cd part2
python main.py
```

**Outputs in `part2/results/`:**

| File | Description |
|------|-------------|
| `S1-{1,2,3}.png` | 3D structure before metric upgrade (3 viewpoints) |
| `S2-{1,2,3}.png` | 3D structure after metric upgrade via Q (3 viewpoints) |
| `k{frame:08d}.png` | Side-by-side: observed vs. reprojected keypoints per frame |
| `mse.png` | Per-frame MSE plot across all 101 frames |

**Console output:** Per-frame MSE (101 values).

---

## Implementation Notes

### Part 1: `camera.py`

**`fit_fundamental_unnormalized(matches)`**
- Builds the 8-point linear system `A @ f = 0` where each row of A encodes one epipolar constraint `x2ᵀ F x1 = 0`.
- Solves via SVD (last right singular vector), then enforces rank-2 by zeroing the smallest singular value of F.
- Scales result so F₃₃ = 1.

**`fit_fundamental_normalized(matches)`**
- Normalizes each image's point set: translate centroid to origin, scale so mean distance = √2.
- Builds and solves the same 8-point system on normalized points.
- Denormalizes: `F = T₂ᵀ @ F_norm @ T₁`.
- Enforces rank-2 and scales as above.

**`camera_calibration(pts_3d, pts_2d)`**
- Sets up the DLT (Direct Linear Transform) system: two linear equations per point pair.
- Solves `A @ p = 0` via SVD; reshapes the last right singular vector into a 3×4 matrix P.

**`triangulation(matches, proj1, proj2)`**
- For each match, calls `triangulation_single`.
- Builds a 4×4 system from the cross-product form of the projection equations.
- Solves via SVD; dehomogenizes the last right singular vector.

### Part 2: `affine.py`

**`normalize_measurements(D)`**
- For each frame i (rows 2i, 2i+1), subtracts the centroid (mean x and mean y) across all N tracked points.
- Returns the centered matrix and per-frame centroids for later reconstruction.

**`get_structure_and_motion(D, k=3)`**
- Computes the full SVD of D (2M × N).
- Truncates to rank k=3: `D ≈ Uₖ Σₖ Vₖᵀ`.
- Splits singular values evenly: `M̂ = Uₖ √Σₖ`, `Ŝ = √Σₖ Vₖᵀ`.

**`get_Q(M)`**
- Solves for the symmetric matrix L = QQᵀ using the metric constraints (orthonormality of camera row vectors) as a linear system in the 6 unique entries of L.
- Recovers Q via Cholesky decomposition of L (with eigenvalue clamping for numerical stability).
