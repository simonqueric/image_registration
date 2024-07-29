import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import pystrum
# import neurite as ne
from monai.losses import (
    GlobalMutualInformationLoss,
    BendingEnergyLoss,
    DiffusionLoss)
from monai.transforms import BorderPad



class Deformation(nn.Module):
    """Class implementing affine deformation of an image (rotation, small shear and small translation)"""
    
    def __init__(self, angle=np.pi/4, dx=0.05, dy=-0.05):
        super(Deformation, self).__init__()
    
        self.angle = angle
        self.affine_transform = torch.Tensor( 
                                    [[[0.99*np.cos(self.angle), -np.sin(self.angle), dx],
                                       [np.sin(self.angle), 0.9*np.cos(self.angle), dy]]])
    
    def forward(self, x):
        b, c, h, w = x.shape
        grid = F.affine_grid(self.affine_transform, size=(b, c, h, w), align_corners=True)
        warped = F.grid_sample(x, grid, mode="bilinear", align_corners=True)
        return warped, grid
    
    
    
class AffineRegistration(nn.Module):
    """Class implementing affine registration of an image (rotation, small shear and small translation)"""
    
    def __init__(self):
        super(AffineRegistration, self).__init__()
        
        self.affine_transform = nn.Parameter(torch.Tensor(
                                        [[[1, 0, 0.],
                                          [0, 1, 0.]]]))

        print(self.affine_transform)
    
    def forward(self, x):
        b, c, h, w = x.shape
        grid = F.affine_grid(self.affine_transform, size=(b, c, h, w), align_corners=True)
        warped = F.grid_sample(x, grid, mode="bilinear", align_corners=True)
        return warped, grid
    
    def fit(self, x):
        pass