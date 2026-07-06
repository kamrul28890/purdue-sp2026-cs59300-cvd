"""
camera.py - Part 1: Fundamental Matrix Estimation, Camera Calibration, Triangulation
CS59300-CVD Assignment 4
"""

import torch
from torch import Tensor


def fit_fundamental_unnormalized(matches: Tensor) -> Tensor:
    r"""
    Estimate the Fundamental matrix using the unnormalized 8-point algorithm.

    The epipolar constraint is:  p2.T @ F @ p1 = 0
    where p1, p2 are homogeneous 2D points in image 1 and image 2.

    Expanding: x2*x1*F11 + x2*y1*F12 + x2*F13
             + y2*x1*F21 + y2*y1*F22 + y2*F23
             +    x1*F31 +    y1*F32 +    F33 = 0

    This gives one linear constraint per match. We form the system Af = 0
    where f = vec(F) and solve via SVD (take last right singular vector).
    Rank-2 constraint is enforced by zeroing the smallest singular value of F.

    Arguments
    - `matches`: [num_matches, 4] — (x1, y1, x2, y2) correspondences

    Return
    - `F`: [3, 3] fundamental matrix (rank-2)
    """
    x1 = matches[:, 0]
    y1 = matches[:, 1]
    x2 = matches[:, 2]
    y2 = matches[:, 3]

    # Build the constraint matrix A such that A @ f = 0
    # Each row: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    ones = torch.ones_like(x1)
    A = torch.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1,      y1,      ones
    ], dim=1)  # [N, 9]

    # Solve A @ f = 0 via SVD; solution is last right singular vector
    _, _, Vt = torch.linalg.svd(A)
    f = Vt[-1]  # [9]
    F = f.reshape(3, 3)

    # Enforce rank-2 constraint by zeroing the smallest singular value
    U, S, Vt2 = torch.linalg.svd(F)
    S_rank2 = S.clone()
    S_rank2[-1] = 0.0
    F = U @ torch.diag(S_rank2) @ Vt2

    # Scale so that F[2,2] = 1
    F = F / F[2, 2]
    return F


def fit_fundamental_normalized(matches: Tensor) -> Tensor:
    r"""
    Estimate the Fundamental matrix using the normalized 8-point algorithm.

    Normalization improves numerical conditioning. For each image:
      1. Translate so the centroid of all points is at the origin.
      2. Scale so the average distance to the origin is sqrt(2).

    The transformation matrix T is:
        T = [[s, 0, -s*cx],
             [0, s, -s*cy],
             [0, 0,     1]]
    where (cx, cy) is the centroid and s = sqrt(2) / mean_dist.

    After computing F on normalized points, denormalize:
        F_orig = T2.T @ F_norm @ T1

    Arguments
    - `matches`: [num_matches, 4] — (x1, y1, x2, y2)

    Return
    - `F`: [3, 3] fundamental matrix (rank-2, denormalized)
    """

    def normalize_points(pts):
        """Normalize 2D points; return (pts_norm_homo, T)."""
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        shifted = pts - torch.stack([cx, cy])
        mean_dist = torch.sqrt((shifted ** 2).sum(dim=1)).mean()
        s = (2.0 ** 0.5) / mean_dist
        T = torch.tensor([
            [s,  0, -s * cx],
            [0,  s, -s * cy],
            [0,  0,       1]
        ], dtype=pts.dtype)
        # Apply transform to get normalized homogeneous coords
        ones = torch.ones(len(pts), 1, dtype=pts.dtype)
        pts_h = torch.cat([pts, ones], dim=1)  # [N, 3]
        pts_norm = (T @ pts_h.t()).t()          # [N, 3]
        return pts_norm, T

    pts1 = matches[:, :2]
    pts2 = matches[:, 2:]

    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build normalized constraint matrix
    x1n, y1n = pts1_norm[:, 0], pts1_norm[:, 1]
    x2n, y2n = pts2_norm[:, 0], pts2_norm[:, 1]
    ones = torch.ones_like(x1n)

    A = torch.stack([
        x2n * x1n, x2n * y1n, x2n,
        y2n * x1n, y2n * y1n, y2n,
        x1n,       y1n,       ones
    ], dim=1)

    _, _, Vt = torch.linalg.svd(A)
    f = Vt[-1]
    F_norm = f.reshape(3, 3)

    # Enforce rank-2
    U, S, Vt2 = torch.linalg.svd(F_norm)
    S[-1] = 0.0
    F_norm = U @ torch.diag(S) @ Vt2

    # Denormalize: F = T2.T @ F_norm @ T1
    F = T2.t() @ F_norm @ T1

    # Scale so F[2,2] = 1
    F = F / F[2, 2]
    return F


def camera_calibration(pts_3d: Tensor, pts_2d: Tensor) -> Tensor:
    r"""
    Estimate the camera projection matrix P (3x4) via linear least squares.

    The projection model (in homogeneous coordinates):
        lambda * [u, v, 1].T = P @ [X, Y, Z, 1].T

    Expanding and eliminating lambda gives two linear equations per point:
        X*P11 + Y*P12 + Z*P13 + P14 - u*X*P31 - u*Y*P32 - u*Z*P33 - u*P34 = 0
        X*P21 + Y*P22 + Z*P23 + P24 - v*X*P31 - v*Y*P32 - v*Z*P33 - v*P34 = 0

    Stack all points into matrix A (size [2N, 12]) and solve A @ p = 0 via SVD.

    Arguments
    - `pts_3d`: [num_points, 3]
    - `pts_2d`: [num_points, 2]

    Return
    - `projection`: [3, 4]
    """
    N = len(pts_3d)
    X, Y, Z = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
    u, v = pts_2d[:, 0], pts_2d[:, 1]
    zeros = torch.zeros(N, dtype=pts_3d.dtype)
    ones = torch.ones(N, dtype=pts_3d.dtype)

    # Each 3D point contributes 2 rows to A
    row_u = torch.stack([X, Y, Z, ones, zeros, zeros, zeros, zeros,
                         -u * X, -u * Y, -u * Z, -u], dim=1)
    row_v = torch.stack([zeros, zeros, zeros, zeros, X, Y, Z, ones,
                         -v * X, -v * Y, -v * Z, -v], dim=1)

    A = torch.cat([row_u, row_v], dim=0)  # [2N, 12]

    _, _, Vt = torch.linalg.svd(A)
    p = Vt[-1]  # [12]
    P = p.reshape(3, 4)
    return P


def triangulation(matches: Tensor, proj1: Tensor, proj2: Tensor) -> Tensor:
    """
    Triangulate 3D points from 2D correspondences and two projection matrices.

    For each correspondence (x1,y1) <-> (x2,y2), solve the linear system
    derived from the projection equations:
        x1 = P1[0] @ X / P1[2] @ X   =>  x1 * (P1[2] @ X) - P1[0] @ X = 0
        y1 = P1[1] @ X / P1[2] @ X   =>  y1 * (P1[2] @ X) - P1[1] @ X = 0
        (similarly for x2, y2 and P2)

    Solve via SVD; the 3D point X = last right singular vector (homogeneous),
    then dehomogenize.

    Arguments
    - `matches`: [num_points, 4] — (x1, y1, x2, y2)
    - `proj1`: [3, 4]
    - `proj2`: [3, 4]

    Return
    - `X`: [N, 3]
    """
    points3d = []
    for i in range(len(matches)):
        x1, y1, x2, y2 = matches[i]
        pt = triangulation_single(x1, y1, x2, y2, proj1, proj2)
        points3d.append(pt)
    return torch.stack(points3d, dim=0)


def triangulation_single(
    x1: Tensor, y1: Tensor,
    x2: Tensor, y2: Tensor,
    P1: Tensor, P2: Tensor
) -> Tensor:
    """
    Triangulate a single 3D point from two 2D correspondences.

    Build a 4x4 linear system A @ X_h = 0 where X_h is homogeneous 3D:
        row0: x1 * P1[2] - P1[0]
        row1: y1 * P1[2] - P1[1]
        row2: x2 * P2[2] - P2[0]
        row3: y2 * P2[2] - P2[1]

    Arguments
    - x1, y1: scalar tensors — point in image 1
    - x2, y2: scalar tensors — point in image 2
    - P1, P2: [3, 4] projection matrices

    Return
    - X: [3] — 3D point (inhomogeneous)
    """
    A = torch.stack([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ], dim=0)  # [4, 4]

    _, _, Vt = torch.linalg.svd(A)
    X_h = Vt[-1]  # [4] homogeneous
    X = X_h[:3] / X_h[3]  # dehomogenize
    return X
