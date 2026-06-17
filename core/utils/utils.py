import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
import os
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from packaging import version
import torchgeometry as tgm

def load_ckpt(model, args, distributed=False, optimizer=None, scheduler=None):
    """ Load checkpoint """
    path = args.restore_ckpt
    ifstrict =args.strict_load
    if not os.path.isfile(path):
        raise ValueError('The checkpoint that you chose does not exist, {}'.format(path))
    checkpoint_dict = torch.load(path, map_location=torch.device('cpu'),weights_only=False)
    # # 临时处理错误保存为多卡的模型参数的，去掉module这几个字
    # mode = 0
    # if mode == 1:
    #     new_dict = {}
    #     for k, v in checkpoint_dict.items():
    #         new_dict[k[7:]] = v
    #     checkpoint_dict = new_dict
    removeflag = 0
    for k, v in checkpoint_dict.items():
        if k[:6] == 'module':
            removeflag = 1
            break
    if removeflag == 1:
        new_dict = {}
        for k, v in checkpoint_dict.items():
            new_dict[k[7:]] = v
        checkpoint_dict = new_dict
    if optimizer is not None and scheduler is not None:
        if 'optimizer' in checkpoint_dict and 'scheduler' in checkpoint_dict:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            scheduler.load_state_dict(checkpoint_dict['scheduler'])
    if 'state_dict' in checkpoint_dict:
        checkpoint_dict = checkpoint_dict['state_dict']
    removeflag = 0
    for k, v in checkpoint_dict.items():
        if k[:9] == 'coarsenet':
            removeflag = 1
            break
    if removeflag == 1:
        new_dict = {}
        for k, v in checkpoint_dict.items():
            new_dict[k[10:]] = v
        checkpoint_dict = new_dict

    if distributed == False:
        if hasattr(model, 'coarsenet'):
            model.coarsenet.load_state_dict(checkpoint_dict, strict=ifstrict)
        else:
            model.load_state_dict(checkpoint_dict, strict=ifstrict)
    else:
        if hasattr(model.module, 'coarsenet'):
            model.module.coarsenet.load_state_dict(checkpoint_dict, strict=ifstrict)
        else:
            model.module.load_state_dict(checkpoint_dict, strict=ifstrict)

def resize_data(img1, img2, flow, factor=1.0):
    _, _, h, w = img1.shape
    h = int(h * factor)
    w = int(w * factor)
    img1 = F.interpolate(img1, (h, w), mode='area')
    img2 = F.interpolate(img2, (h, w), mode='area')
    flow = F.interpolate(flow, (h, w), mode='area') * factor
    return img1, img2, flow

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def transform(T, p):
    assert T.shape == (4,4)
    return np.einsum('H W j, i j -> H W i', p, T[:3,:3]) + T[:3, 3]

def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum('H W, H W j, i j -> H W i', depth1, img_1_coords, np.linalg.inv(K1))
    rel_pose = np.linalg.inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum('H W j, i j -> H W i', cam2_coords, K2))

def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data['T0'], data['T1'], data['K0'], data['K1'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords1 = reproject(depth1, data['T1'], data['T0'], data['K1'], data['K0'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0
    
    return flow_01, flow_10

def check_cycle_consistency(flow_01, flow_10):
    flow_01 = torch.from_numpy(flow_01).permute(2, 0, 1)[None]
    flow_10 = torch.from_numpy(flow_10).permute(2, 0, 1)[None]
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < 0.1 * min(H, W)).float()
    return mask[0].numpy()

def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (b, h, w)
        ],
        indexing = 'ij'
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid

def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype

# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, mode=InterpolationMode.BILINEAR))
    return TupleCompose(ops)

def get_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    if normalize:
        ops.append(TupleToTensorScaled())
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    else:
        if unscale:
            ops.append(TupleToTensorUnscaled())
        else:
            ops.append(TupleToTensorScaled())
    return TupleCompose(ops)

class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode)

    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        return [self.normalize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode="bilinear")[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode="bilinear"
    )[:, 0, :, 0]
    consistent_mask = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs() < 0.05
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0

@torch.no_grad()
def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    Args:
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if version.parse(torch.__version__) >= version.parse("1.3"):
        # to be consistent to old version, I put align_corners=True.
        # to investigate if align_corners False is better.
        output = F.grid_sample(x, vgrid, align_corners=True)
    else:
        output = F.grid_sample(x, vgrid)
    return output

@torch.no_grad()
def torchPSNR(prd_img, tar_img):
	if not isinstance(prd_img, torch.Tensor):
		prd_img = torch.from_numpy(prd_img)
		tar_img = torch.from_numpy(tar_img)

	imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
	rmse = (imdff**2).mean().sqrt()
	ps = 20 * torch.log10(1/rmse)
	return ps

def disp_to_coords(four_point, coords):
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, 1]) 
    four_point_org[:, 1, 1] = torch.Tensor([1, 1]) #[W, H]

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(coords.shape[0], 1, 1, 1)
    four_point_new = four_point_org + four_point
    four_point_new[:, :, :, 0] *= coords.shape[3]-1
    four_point_new[:, :, :, 1] *= coords.shape[2]-1
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
    H = tgm.get_perspective_transform(four_point_org, four_point_new)
    gridy, gridx = torch.meshgrid(torch.linspace(0, coords.shape[3]-1, steps=coords.shape[3]), torch.linspace(0, coords.shape[2]-1, steps=coords.shape[2]))
    points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, coords.shape[3] * coords.shape[2]))),
                       dim=0).unsqueeze(0).repeat(coords.shape[0], 1, 1).to(four_point.device)
    points_new = H.bmm(points)
    points_new = points_new / points_new[:, 2, :].unsqueeze(1)
    points_new = points_new[:, 0:2, :]
    coords = torch.cat((points_new[:, 0, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1),
                      points_new[:, 1, :].reshape(coords.shape[0], coords.shape[3], coords.shape[2]).unsqueeze(1)), dim=1)
    return coords

def HomographyGridGen(four_points, coords):
    """Dense correspondence map generator, corresponding to a homography transform."""
    # create grid in numpy
    # self.grid = np.zeros( [self.out_h, self.out_w, 3], dtype=np.float32)
    # sampling grid with dim-0 coords (Y)
    b,_, out_h, out_w = coords.shape
    grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
    # grid_X,grid_Y: size [1,H,W,1,1]
    grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3).to(four_points.device)
    grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3).to(four_points.device)

    H = homography_mat_from_4_pts(four_points)
    h0 = H[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h1 = H[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h2 = H[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h3 = H[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h4 = H[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h5 = H[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h6 = H[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h7 = H[:, 7].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    h8 = H[:, 8].unsqueeze(1).unsqueeze(2).unsqueeze(3)

    grid_X = expand_dim(grid_X, 0, b)
    grid_Y = expand_dim(grid_Y, 0, b)

    grid_Xp = grid_X * h0 + grid_Y * h1 + h2
    grid_Yp = grid_X * h3 + grid_Y * h4 + h5
    k = grid_X * h6 + grid_Y * h7 + h8

    grid_Xp /= k
    grid_Yp /= k

    mapping = torch.cat((grid_Xp, grid_Yp), 3)
    return mapping

def homography_mat_from_4_pts(theta):
    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()

    xp = theta[:, :4].unsqueeze(2);
    yp = theta[:, 4:].unsqueeze(2)

    x = torch.FloatTensor([-1, -1, 1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    y = torch.FloatTensor([-1, 1, -1, 1]).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    z = torch.zeros(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    o = torch.ones(4).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    single_o = torch.ones(1).unsqueeze(1).unsqueeze(0).expand(b, 1, 1)

    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()

    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
                   torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    # find homography by assuming h33 = 1 and inverting the linear system
    h = torch.bmm(torch.linalg.inv(A[:, :, :8]), -A[:, :, 8].unsqueeze(2))
    # add h33
    h = torch.cat([h, single_o], 1)

    H = h.squeeze(2)

    return H

def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))