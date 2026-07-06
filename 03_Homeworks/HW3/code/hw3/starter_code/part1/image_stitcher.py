"""Implements image stitching."""

import numpy as np
import torch
import torch.nn.functional as F
import kornia
from helpers import plot_inlier_matches, compute_harris_response, get_harris_points
import matplotlib.pyplot as plt

from skimage.transform import ProjectiveTransform, warp


class ImageStitcher(object):
  def __init__(self, img1, img2, keypoint_type='harris', descriptor_type='pixel'):
    """
    Inputs:
        img1: h x w tensor.
        img2: h x w tensor.
        keypoint_type: string in ['harris']
        descriptor_type: string in ['pixel', 'hynet']
    """
    self.img1 = img1
    self.img2 = img2
    self.keypoint_type = keypoint_type
    self.descriptor_type = descriptor_type
    #### Your Implementation Below #### 

    # Extract keypoints
    self.keypoints1 = None 
    self.keypoints2 = None 

    # Extract descriptors at each keypoint
    self.desc1 = None 
    self.desc2 = None 

    # Compute putative matches and match the keypoints.
    matches = None 
    matched_keypoints = None 

    # Perform RANSAC to find the best homography and inliers
    inliers = None

    # Plot the inliers
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_inlier_matches(ax,
                        kornia.utils.tensor_to_image(img1),
                        kornia.utils.tensor_to_image(img2),
                        matched_keypoints[inliers])
    plt.savefig('inlier_matches_%s.png' % self.descriptor_type)

    # Refit with all inliers to get the final homography
    final_homography = None 
    stitched = None 

    plt.figure()
    plt.imshow(stitched)
    plt.gray()
    plt.savefig('stitched_%s.png' % self.descriptor_type)

  def _get_keypoints(self, img):
    """
    Extract keypoints from the image.

    Inputs:
        img: h x w tensor.
    Outputs:
        keypoints: N x 2 numpy array.
    """
    keypoints = None 
    return keypoints

  def _get_descriptors(self, img, keypoints):
    """
    Extract descriptors from the image at the given keypoints.

    Inputs:
        img: h x w tensor.
        keypoints: N x 2 tensor.
    Outputs:
        descriptors: N x D tensor.
    """
    if self.descriptor_type == 'pixel':
      descriptors = None 
    elif self.descriptor_type == 'hynet':
      descriptors = None 
    return descriptors

  def _get_putative_matches(self, desc1, desc2, max_num_matches=100):
    """
    Compute putative matches between two sets of descriptors.

    Inputs:
        desc1: N x D tensor.
        desc2: M x D tensor.
        max_num_matches: Integer
    Outputs:
        matches: 2 x max_num_matches tensor.
    """
    matches = None 
    return matches

  def _get_homography(self, matched_keypoints):
    """
    Compute the homography between two images.

    Inputs:
        matched_keypoints: N x 4 tensor.
    Outputs:
        homography: 3 x 3 tensor.
    """
    homography = None 
    return homography

  def _homography_inliers(self, H, matched_keypoints, inlier_threshold=20):
    """
    Compute the inliers for the given homography.

    Inputs:
        H: Homography 3 x 3 tensor.
        matched_keypoints: N x 4 tensor.
        inlier_threshold: upper bounds on what counts as inlier.
    Outputs:
        inliers: N tensor, indicates whether each matched keypoint is an inlier.
    """
    inliers = None 
    return inliers

  def _ransac(self, matched_keypoints, num_iterations=30000, inlier_threshold=20):
    """
    Perform RANSAC to find the best homography.

    Inputs:
      matched_keypoints: N x 4 tensor.
      num_iterations: Number of iteration to run RANSAC.
      inlier_threshold: upper bounds on what counts as inlier.
    Outputs:
      best_inliers: N tensor, indicates whether each matched keypoint is an inlier.
      best_homography: 3 x 3 tensor
    """
    best_inliers = None
    best_homography = None
    return best_inliers, best_homography

  def stitch(self, final_homography):
    """
    Stitch the two images together.

    Inputs:
        final_homography: 3 x 3 tensor.
    Outputs:
        stitched: h x w tensor.
    """
    stitched = None 
    return stitched
