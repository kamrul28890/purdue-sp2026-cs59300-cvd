"""
affine.py - Part 2: Affine Factorization (Tomasi-Kanade)
CS59300-CVD Assignment 4
"""

import torch
from torch import Tensor


def normalize_measurements(X: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Normalize the 2M x N measurement matrix by subtracting the centroid
    of each view's image points.

    The measurement matrix has the form:
        x1(1)  x2(1)  ...  xN(1)
        y1(1)  y2(1)  ...  yN(1)
        ...
        x1(M)  x2(M)  ...  xN(M)
        y1(M)  y2(M)  ...  yN(M)

    For each frame i (rows 2i and 2i+1), we compute the centroid of
    all N tracked points and subtract it from those two rows.

    This centers the data, which is required by the Tomasi-Kanade
    factorization to remove the translation component.

    Arguments
    - `X`: [2M, N] measurement matrix

    Return
    - `D`: [2M, N] normalized measurement matrix (zero-mean per view)
    - `mean`: [M, 2] per-view centroid, where mean[i] = (cx_i, cy_i).
              Shape is stored for convenient re-addition in main.py.
    """
    two_M, N = X.shape
    M = two_M // 2

    D = X.clone()
    mean_list = []  # store as list, will convert to tensor

    for i in range(M):
        # x-coordinates for frame i
        cx = X[2 * i].mean()
        # y-coordinates for frame i
        cy = X[2 * i + 1].mean()

        D[2 * i]     = X[2 * i]     - cx
        D[2 * i + 1] = X[2 * i + 1] - cy

        # Store centroid as a (2, 1) column vector to match D's structure
        mean_list.append(torch.stack([cx.expand(1), cy.expand(1)], dim=0))  # [2, 1]

    # mean[i] is shape [2, 1]; used in main.py as mean[i] added back to keypoints
    # main.py uses: keypoints1 = D[2*i:2*i+2] + mean[i]
    # so mean[i] should broadcast over [2, N] — keep [2, 1] shape
    mean = torch.stack(mean_list, dim=0)  # [M, 2, 1]

    return D, mean


def get_structure_and_motion(
    D: Tensor,
    k: int = 3
) -> tuple[Tensor, Tensor]:
    r"""
    Recover the motion matrix M (2M x 3) and structure matrix S (3 x N)
    from the normalized measurement matrix D (2M x N) via truncated SVD.

    By the Tomasi-Kanade theorem, under affine projection the measurement
    matrix is (approximately) rank 3. We factor it as:

        D ≈ M_hat @ S_hat

    using the rank-3 approximation:
        D ≈ U_k @ diag(sigma_k) @ Vt_k

    where U_k is [2M, 3], sigma_k is [3], Vt_k is [3, N].

    We split the singular values evenly between M and S:
        M_hat = U_k @ diag(sqrt(sigma_k))       [2M, 3]
        S_hat = diag(sqrt(sigma_k)) @ Vt_k      [3, N]

    Arguments
    - `D`: [2M, N] normalized measurement matrix
    - `k`: rank of approximation (default 3 for affine SfM)

    Return
    - `M_hat`: [2M, 3] motion matrix
    - `S_hat`: [3, N] structure matrix
    """
    U, sigma, Vt = torch.linalg.svd(D, full_matrices=False)

    # Keep only top-k components
    U_k   = U[:, :k]          # [2M, k]
    sig_k = sigma[:k]          # [k]
    Vt_k  = Vt[:k, :]         # [k, N]

    sqrt_sig = torch.diag(torch.sqrt(sig_k))  # [k, k]

    M_hat = U_k @ sqrt_sig    # [2M, 3]
    S_hat = sqrt_sig @ Vt_k   # [3, N]

    return M_hat, S_hat


def get_Q(M: Tensor) -> Tensor:
    r"""
    Compute the affine ambiguity correction matrix Q (3x3) such that:
        M_corrected = M @ Q
        S_corrected = Q^{-1} @ S

    Under the affine camera model each pair of rows (2i, 2i+1) of M
    represents the camera's x- and y-row vectors (i_i, j_i). The metric
    constraints require:

        i_i . i_i = 1   (unit x-axis)
        j_i . j_j = 1   (unit y-axis)
        i_i . j_i = 0   (orthogonality)

    Letting L = Q @ Q.T (symmetric positive definite 3x3), these become:

        i_i @ L @ i_i.T = 1
        j_i @ L @ j_i.T = 1
        i_i @ L @ j_i.T = 0

    We pack L as a 6-vector l = [L11, L12, L13, L22, L23, L33] and
    solve the linear system G @ l = c (with c = [1, 1, 0] per frame).

    Then we recover Q via Cholesky decomposition of L:
        L = C @ C.T  =>  Q = C

    Arguments
    - `M`: [2M, 3] motion matrix from get_structure_and_motion

    Return
    - `Q`: [3, 3] correction matrix
    """
    two_M = M.shape[0]
    num_frames = two_M // 2

    # Build the linear system for the symmetric matrix L
    # l = [L11, L12, L13, L22, L23, L33]
    # For rows i_i = M[2i, :] and j_i = M[2i+1, :]:
    #   i_i L i_i.T = 1  =>  coeff vector from outer product
    #   j_i L j_i.T = 1
    #   i_i L j_i.T = 0

    rows = []
    rhs  = []

    def sym_coeff(a, b):
        """Coefficient vector for a.T @ L @ b, where L is symmetric 3x3.
        l = [L11, L12, L13, L22, L23, L33]
        a.T L b = a0*b0*L11 + (a0*b1+a1*b0)*L12 + (a0*b2+a2*b0)*L13
                + a1*b1*L22 + (a1*b2+a2*b1)*L23 + a2*b2*L33
        """
        c = torch.zeros(6, dtype=M.dtype)
        c[0] = a[0] * b[0]
        c[1] = a[0] * b[1] + a[1] * b[0]
        c[2] = a[0] * b[2] + a[2] * b[0]
        c[3] = a[1] * b[1]
        c[4] = a[1] * b[2] + a[2] * b[1]
        c[5] = a[2] * b[2]
        return c

    for i in range(num_frames):
        ii = M[2 * i]       # i-vector for frame i
        ji = M[2 * i + 1]   # j-vector for frame i

        rows.append(sym_coeff(ii, ii))  # ii L ii.T = 1
        rhs.append(torch.tensor(1.0, dtype=M.dtype))

        rows.append(sym_coeff(ji, ji))  # ji L ji.T = 1
        rhs.append(torch.tensor(1.0, dtype=M.dtype))

        rows.append(sym_coeff(ii, ji))  # ii L ji.T = 0
        rhs.append(torch.tensor(0.0, dtype=M.dtype))

    G = torch.stack(rows, dim=0)                    # [3*M, 6]
    c = torch.stack(rhs,  dim=0).unsqueeze(1)       # [3*M, 1]

    # Solve G @ l = c in least-squares sense
    l, _, _, _ = torch.linalg.lstsq(G, c)
    l = l.squeeze()  # [6]

    # Reconstruct symmetric L from l
    L = torch.zeros(3, 3, dtype=M.dtype)
    L[0, 0] = l[0]
    L[0, 1] = L[1, 0] = l[1]
    L[0, 2] = L[2, 0] = l[2]
    L[1, 1] = l[3]
    L[1, 2] = L[2, 1] = l[4]
    L[2, 2] = l[5]

    # Ensure positive semi-definiteness (clamp small negative eigenvalues)
    eigvals, eigvecs = torch.linalg.eigh(L)
    eigvals = torch.clamp(eigvals, min=1e-8)
    L = eigvecs @ torch.diag(eigvals) @ eigvecs.t()

    # Cholesky decomposition: L = C @ C.T  =>  Q = C
    try:
        Q = torch.linalg.cholesky(L)
    except Exception:
        # Fallback: use sqrtm via eigen decomposition
        Q = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.t()

    return Q
