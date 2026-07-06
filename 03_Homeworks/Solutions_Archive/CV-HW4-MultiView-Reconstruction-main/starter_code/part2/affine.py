import torch
from torch import Tensor

def normalize_measurements(X: Tensor) -> tuple[Tensor, Tensor]:
    r'''
    x1(1)  x2(1)  x3(1) ... xN(1)
    y1(1)  y2(1)  y3(1) ... yN(1)
    ...
    x1(M)  x2(M)  x3(M) ... xN(M)
    y1(M)  y2(M)  y3(M) ... yN(M)
    '''
    num_frames = X.shape[0] // 2
    D = X.clone()
    means = []

    for i in range(num_frames):
        rows = D[2 * i : 2 * i + 2]
        mean = rows.mean(dim=1, keepdim=True)
        D[2 * i : 2 * i + 2] = rows - mean
        means.append(mean)

    return D, torch.stack(means, dim=0)

def get_structure_and_motion(
    D: Tensor, 
    k: int = 3
) -> tuple[Tensor, Tensor]:
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    V_k = Vh[:k, :]

    S_root = torch.diag(torch.sqrt(S_k))
    M = U_k @ S_root
    structure = S_root @ V_k
    return M, structure

def get_Q(M) -> Tensor:
    num_frames = M.shape[0] // 2
    dtype = M.dtype
    device = M.device

    A_rows = []
    b_rows = []

    def coeffs(a: Tensor, b: Tensor) -> Tensor:
        return torch.stack(
            [
                a[0] * b[0],
                a[0] * b[1] + a[1] * b[0],
                a[0] * b[2] + a[2] * b[0],
                a[1] * b[1],
                a[1] * b[2] + a[2] * b[1],
                a[2] * b[2],
            ]
        )

    for i in range(num_frames):
        mx = M[2 * i]
        my = M[2 * i + 1]

        A_rows.append(coeffs(mx, mx))
        b_rows.append(torch.tensor(1.0, dtype=dtype, device=device))

        A_rows.append(coeffs(my, my))
        b_rows.append(torch.tensor(1.0, dtype=dtype, device=device))

        A_rows.append(coeffs(mx, my))
        b_rows.append(torch.tensor(0.0, dtype=dtype, device=device))

    A = torch.stack(A_rows, dim=0)
    b = torch.stack(b_rows, dim=0)

    l = torch.linalg.lstsq(A, b).solution
    L = torch.tensor(
        [
            [l[0], l[1], l[2]],
            [l[1], l[3], l[4]],
            [l[2], l[4], l[5]],
        ],
        dtype=dtype,
        device=device,
    )

    eigvals, eigvecs = torch.linalg.eigh(L)
    eigvals = torch.clamp(eigvals, min=1e-8)
    Q = eigvecs @ torch.diag(torch.sqrt(eigvals))
    return Q
