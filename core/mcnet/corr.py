import torch.nn as nn
import torch.nn.functional as F
from mcnet.flow_utils import *


def local_correlation(fmap1, fmap2, radius=4):
    """Compute local 9x9 correlation with PyTorch ops.

    The old CuPy RawKernel path requires CUDA toolkit headers at runtime.
    This implementation is slower but avoids runtime CUDA compilation.
    """
    batch, channels, height, width = fmap1.shape
    kernel_size = 2 * radius + 1
    fmap2_patches = F.unfold(fmap2, kernel_size=kernel_size, padding=radius)
    fmap2_patches = fmap2_patches.view(batch, channels, kernel_size * kernel_size, height, width)
    corr = (fmap1.unsqueeze(2) * fmap2_patches).sum(dim=1)
    return corr / float(channels)


class LocalCorr:
    def __init__(self, fmap1, fmap2):
        self.map1 = fmap1
        self.map2 = fmap2
        self.N, self.C, self.H, self.W = fmap1.shape
        self.coords = coords_grid(self.N, self.H, self.W).to(fmap1.device)

    def warp(self, coords, image, h, w):
        coords[: ,0 ,: ,:] = 2.0 *coords[: ,0 ,: ,:].clone() / max(self.W -1 ,1 ) -1.0
        coords[: ,1 ,: ,:] = 2.0 *coords[: ,1 ,: ,:].clone() / max(self.H -1 ,1 ) -1.0

        coords = coords.permute(0 ,2 ,3 ,1)
        output = F.grid_sample(image, coords, align_corners=True, padding_mode="border")
        return output

    def __call__(self, coords):
        map2_warp = self.warp(coords, self.map2, self.H, self.W)
        return local_correlation(self.map1, map2_warp, radius=4)
    
