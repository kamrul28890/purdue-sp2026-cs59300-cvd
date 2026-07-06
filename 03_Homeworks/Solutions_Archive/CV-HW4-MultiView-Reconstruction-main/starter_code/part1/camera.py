import torch
from torch import Tensor


def _enforce_rank2(F: Tensor) -> Tensor:
    U, S, Vh = torch.linalg.svd(F)
    S[-1] = 0.0
    return U @ torch.diag(S) @ Vh


def _fit_fundamental_linear(matches: Tensor) -> Tensor:
    x1 = matches[:, 0]
    y1 = matches[:, 1]
    x2 = matches[:, 2]
    y2 = matches[:, 3]

    A = torch.stack(
        [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            torch.ones_like(x1),
        ],
        dim=1,
    )

    _, _, Vh = torch.linalg.svd(A)
    F = Vh[-1].reshape(3, 3)
    return _enforce_rank2(F)


def _normalize_points_2d(points: Tensor) -> tuple[Tensor, Tensor]:
    centroid = points.mean(dim=0)
    centered = points - centroid
    dist = torch.linalg.norm(centered, dim=1)
    mean_dist = dist.mean().clamp_min(1e-12)
    scale = torch.sqrt(torch.tensor(2.0, dtype=points.dtype, device=points.device)) / mean_dist

    T = torch.tensor(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=points.dtype,
        device=points.device,
    )

    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)
    points_n = (points_h @ T.t())[:, :2]
    return points_n, T

def fit_fundamental_unnormalized(matches: Tensor) -> Tensor:
    r'''
    Fundamental matrix using unnormalized algorithm
    
    Arguments
    - `matches`: [`num_matches`, `4`]
    e.g. (x1, y1, x2, y2)
    the first two numbers is a point in the first image
    the last two numbers is a point in the second image
    
    Return
    - `F`: [`3`, `3`]
    '''
    F = _fit_fundamental_linear(matches)
    if torch.abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]
    return F

def fit_fundamental_normalized(matches: Tensor) -> Tensor:
    r'''
    Fundamental matrix using normalized algorithm
    
    Arguments
    - `matches`: [`num_matches`, 4]
    e.g. (x1, y1, x2, y2)
    the first two numbers is a point in the first image
    the last two numbers is a point in the second image
    
    Return
    - `F`: [3, 3]
    '''
    pts1_n, T1 = _normalize_points_2d(matches[:, :2])
    pts2_n, T2 = _normalize_points_2d(matches[:, 2:])

    matches_n = torch.cat([pts1_n, pts2_n], dim=1)
    F_n = _fit_fundamental_linear(matches_n)

    F = T2.t() @ F_n @ T1
    if torch.abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]
    return F

def camera_calibration(pts_3d: Tensor, pts_2d: Tensor) -> Tensor:
    r"""
    Camera Calibration

    Arguments
    - `pts_3d`: [`num_points`, `3`]
    - `pts_2d`: [`num_points`, `2`]

    Return
    - `projection`: [`3`, `4`]
    """
    n = pts_3d.shape[0]
    ones = torch.ones((n, 1), dtype=pts_3d.dtype, device=pts_3d.device)
    X = torch.cat([pts_3d, ones], dim=1)
    x = pts_2d[:, 0]
    y = pts_2d[:, 1]

    O = torch.zeros_like(X)
    A1 = torch.cat([O, -X, y[:, None] * X], dim=1)
    A2 = torch.cat([X, O, -x[:, None] * X], dim=1)
    A = torch.cat([A1, A2], dim=0)

    _, _, Vh = torch.linalg.svd(A)
    P = Vh[-1].reshape(3, 4)

    if torch.abs(P[2, 3]) > 1e-12:
        P = P / P[2, 3]
    return P

def triangulation(matches, proj1, proj2) -> Tensor:
    """
    Triangulation

    Arguments
    - `matches`: [`num_points`, `4`]
    - `proj1`: [`3`, `4`]
    - `proj2`: [`3`, `4`]

    Return
    - `X`: [`N`, `3`]
    """
    points = []
    for x1, y1, x2, y2 in matches:
        points.append(triangulation_single(x1, y1, x2, y2, proj1, proj2))
    return torch.stack(points, dim=0)

def triangulation_single(x1, y1, x2, y2, P1, P2) -> Tensor:
    """
    Return
    - `X`: [`3`]
    """
    A = torch.stack(
        [
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1],
        ],
        dim=0,
    )
    _, _, Vh = torch.linalg.svd(A)
    Xh = Vh[-1]
    w = Xh[3]
    if torch.abs(w) < 1e-12:
        w = torch.where(w >= 0, torch.tensor(1e-12, dtype=w.dtype, device=w.device), torch.tensor(-1e-12, dtype=w.dtype, device=w.device))
    X = Xh[:3] / w
    return X
