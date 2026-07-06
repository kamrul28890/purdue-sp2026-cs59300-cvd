"""Implements the alignment algorithm."""

import torch
import torchvision
from metrics import ncc, mse, ssim
from helpers import custom_shifts


class AlignmentModel:
  def __init__(self, image_name, metric='ssim', padding='circular'):
    # Image name
    self.image_name = image_name
    # Metric to use for alignment
    self.metric = metric
    # Padding mode for custom_shifts
    self.padding = padding

  def save(self, output_name):
    torchvision.utils.save_image(self.rgb, output_name)

  def align(self):
    """Aligns the image using the metric specified in the constructor.
       Experiment with the ordering of the alignment.

       Finally, outputs the rgb image in self.rgb.
    """
    self.img = self._load_image()
    self.rgb = self.img  # TODO: Replace this with the aligned RGB image

    ## Your alignment code here ##


  def save(self, output_name):
    torchvision.utils.save_image(self.rgb, output_name)

  def _load_image(self):
    """Load the image from the image_name path,
       typecast it to float, and normalize it.

       Returns: torch.Tensor of shape (H, W)
    """
    ret = None
    ## Your CODE HERE ##

    return ret

  def _crop_and_divide_image(self):
    """Crop the image boundary and divide the image into three parts, padded to the same size.

       Feel free to be creative about this.
       You can eyeball the boundary values, or write code to find approximate cut-offs.
       Hint: Plot out the average values per row / column and visualize it!

       Returns: B, G, R torch.Tensor of shape (roughly H//3, W)
    """
    b_channel = None
    g_channel = None
    r_channel = None 
    ## Your CODE HERE ##


    return b_channel, g_channel, r_channel

  def _align_pairs(self, img1, img2, delta):
    """
    Aligns two images using the metric specified in the constructor.
    Returns: Tuple of (u, v) shifts that minimizes the metric.
    """
    align_idx = (0,0) 
    ## Your CODE HERE ##

    return align_idx
