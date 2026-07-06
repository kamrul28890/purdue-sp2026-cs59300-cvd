import torch
from torch import Tensor

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
    # YOUR CODE HERE
    pass

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
    # YOUR CODE HERE
    pass

def camera_calibration(pts_3d: Tensor, pts_2d: Tensor) -> Tensor:
    r"""
    Camera Calibration

    Arguments
    - `pts_3d`: [`num_points`, `3`]
    - `pts_2d`: [`num_points`, `2`]

    Return
    - `projection`: [`3`, `4`]
    """
    # YOUR CODE HERE
    pass

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
    # YOUR CODE HERE
    pass

def triangulation_single(x1, y1, x2, y2, P1, P2) -> Tensor:
    """
    Return
    - `X`: [`3`]
    """
    # YOUR CODE HERE
    pass
