"""Implements a differentiable alignmnet alg."""

import torch
from alignment_model import AlignmentModel
from metrics import ncc, mse, ssim
from utils.misc_helper import custom_shifts

class DiffAlignment(AlignmentModel):
  def __init__(self, image_name, metric='mse', padding='circular'):
    super().__init__(image_name, metric, padding)

  def align(self):
    self.img = self._load_image()
    self.b, self.g, self.r = self._crop_and_divide_image()
    self.b = self.b.cuda()
    self.g = self.g.cuda()
    self.r = self.r.cuda()
    lr=0.005
    g_idx,_ = self._align_pairs(self.b, self.g, lr=lr)
    print('-------------------')
    r_idx,_ = self._align_pairs(self.b, self.r, lr=lr)
    self.g_aligned = custom_shifts(self.g, g_idx, dims=(0,1), padding=self.padding)
    self.r_aligned = custom_shifts(self.r, r_idx, dims=(0,1), padding=self.padding)
    self.rgb = torch.stack([self.r_aligned, self.g_aligned, self.b], dim=0)

  def _align_pairs(self, img1, img2, lr=0.1):
    if self.metric == 'ncc':
      loss_fn = ncc
    elif self.metric == 'mse':
      loss_fn = mse
    # Create alignment module
    align_net = AlignNet(img1.size()).cuda()

    # Create optimizer
    optimizer = torch.optim.AdamW(align_net.parameters(), lr=lr)

    # Loss function
    for k in range(5000):
      optimizer.zero_grad()
      img2_shifted = align_net(img2)
      loss = loss_fn(img1, img2_shifted)
      loss.backward()
      optimizer.step()
      if k % 1000 == 0:
        shifts = align_net.shifts.detach().cpu().numpy()
        shifts = torch.clamp(align_net.shifts, -0.1, 0.1)
        align_idx = [-int(shifts[1]*(img1.shape[0]/2)), -int(shifts[0]*(img1.shape[1]/2))]
        print(align_idx)
        print(loss.item())
    shifts = torch.clamp(align_net.shifts, -0.1, 0.1)
    align_idx = [-int(shifts[1]*(img1.shape[0]/2)), -int(shifts[0]*(img1.shape[1]/2))]
    return align_idx, align_net


import torch.nn as nn
import torch.nn.functional as F
class AlignNet(nn.Module):
  def __init__(self, img_size):
    super(AlignNet, self).__init__()
    self.img_size = img_size
    # Only have parameters for the shiftsA
    ## Your CODE HERE ##
    self.shift = None

  def forward(self, img):
    ## Your CODE HERE ##
    transformed_img = None
    return transformed_img
