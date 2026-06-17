import torch
import torch.nn.functional as F

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


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def initialize_flow(img, downsample=4):
    N, C, H, W = img.shape
    coords0 = coords_grid(N, H//downsample, W//downsample).to(img.device)
    coords1 = coords_grid(N, H//downsample, W//downsample).to(img.device)

    return coords0, coords1


def get_perspective_transform(src, dst):
    """Batch 4-point perspective transform using torch.linalg.solve.

    This replaces torchgeometry.get_perspective_transform, which calls the
    removed torch.gesv API in newer PyTorch versions.
    """
    if src.shape != dst.shape or src.shape[-2:] != (4, 2):
        raise ValueError(f"Expected src and dst to have shape [B, 4, 2], got {src.shape} and {dst.shape}")

    batch = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    row_x = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    row_y = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)

    A = torch.stack(
        [row_x[:, 0], row_y[:, 0], row_x[:, 1], row_y[:, 1],
         row_x[:, 2], row_y[:, 2], row_x[:, 3], row_y[:, 3]],
        dim=1,
    )
    b = torch.stack([u[:, 0], v[:, 0], u[:, 1], v[:, 1], u[:, 2], v[:, 2], u[:, 3], v[:, 3]], dim=1)

    h = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    H = torch.cat([h, torch.ones(batch, 1, device=src.device, dtype=src.dtype)], dim=1)
    return H.view(batch, 3, 3)


def disp_to_coords(four_point, coords, downsample=4):
    four_point = four_point / downsample
    dtype = four_point.dtype
    device = four_point.device
    height, width = coords.shape[2], coords.shape[3]

    four_point_org = torch.zeros((2, 2, 2), device=device, dtype=dtype)
    four_point_org[:, 0, 0] = torch.tensor([0, 0], device=device, dtype=dtype)
    four_point_org[:, 0, 1] = torch.tensor([width - 1, 0], device=device, dtype=dtype)
    four_point_org[:, 1, 0] = torch.tensor([0, height - 1], device=device, dtype=dtype)
    four_point_org[:, 1, 1] = torch.tensor([width - 1, height - 1], device=device, dtype=dtype)

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(coords.shape[0], 1, 1, 1)
    four_point_new = four_point_org + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
    H = get_perspective_transform(four_point_org, four_point_new)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, height - 1, steps=height, device=device, dtype=dtype),
        torch.linspace(0, width - 1, steps=width, device=device, dtype=dtype),
        indexing='ij',
    )
    points = torch.cat((grid_x.flatten().unsqueeze(0), grid_y.flatten().unsqueeze(0),
                        torch.ones((1, height * width), device=device, dtype=dtype)),
                       dim=0).unsqueeze(0).repeat(coords.shape[0], 1, 1)
    points_new = H.bmm(points)
    points_new = points_new / points_new[:, 2, :].unsqueeze(1)
    points_new = points_new[:, 0:2, :]
    coords = torch.cat((points_new[:, 0, :].reshape(coords.shape[0], height, width).unsqueeze(1),
                        points_new[:, 1, :].reshape(coords.shape[0], height, width).unsqueeze(1)), dim=1)
    return coords
