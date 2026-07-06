import torch
from torch import Tensor

def normalize_measurements(X: Tensor) -> Tensor:
    r'''
    x1(1)  x2(1)  x3(1) ... xN(1)
    y1(1)  y2(1)  y3(1) ... yN(1)
    ...
    x1(M)  x2(M)  x3(M) ... xN(M)
    y1(M)  y2(M)  y3(M) ... yN(M)
    '''
    # YOUR CODE HERE
    pass

def get_structure_and_motion(
    D: Tensor, 
    k: int = 3
) -> tuple[Tensor, Tensor]:
    # YOUR CODE HERE
    pass

def get_Q(M) -> Tensor:
    # YOUR CODE HERE
    pass
