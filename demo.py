import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import ConcatDataset
import random

from config.parser import parse_args

import datasets
from raft import RAFT
from gcsraft_certainty import GCSRAFT_certainty
from orb_sift import ORB, SIFT
from hraft import HomoRAFT

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.visualization_utils import overlay_semantic_mask
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
from utils import frame_utils
from datasets import MegadepthBuilder

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def get_probability_map(info, args, R=1):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    # TODO PDCNet+里有公式（7） 找到两篇论文里log_b和方差var_map的关系 注意是无穷范数
    p_r = torch.sum(weight * (1 - torch.exp(- R/torch.exp(log_b)))**2, dim=1, keepdim=True) 
    return p_r

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final =output['flow'][-1]
    if 'info' in output and output['info'] is not None:
        info_final = output['info'][-1]
    else:
        info_final = None
    B, _, H, W = flow_final.shape
    if 'certainty' in output:
        confidence_map = output['certainty'][-1]
    else:
        confidence_map = torch.ones(size=(B, 1, H, W))
    # if isinstance(model, SEARAFT) or isinstance(model, SwinRAFT): #TODO 不是说他们不能预测可能性，这只是暂时的安排
    #     B, _, H, W = flow_final.shape
    #     confidence_map = torch.ones(size=(B, 1, H, W))
    # else:
    #     confidence_map = output['uncertainty']['inv_cyclic_consistency_error']
    return flow_final, info_final, confidence_map

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt, valid = None, min_confidence = 0.30):
    path = f"demo/{name}/"
    if not os.path.exists(path):
        os.mkdir(path)
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    flow_gt_vis = flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}gt.jpg", flow_gt_vis)
    # flow, info = calc_flow(args, model, image1, image2)
    flow, info, confidence_map = forward_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow_final.jpg", flow_vis)
    diff = flow_gt - flow
    diff_vis = flow_to_image(diff[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}error_final.jpg", diff_vis)
    vis_heatmap(f"{path}certainty.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), confidence_map[0].permute(1, 2, 0).cpu().numpy())
    if info is not None:
        heatmap = get_heatmap(info, args)
        vis_heatmap(f"{path}heatmap_final.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
        probmap = get_probability_map(info, args)
        vis_heatmap(f"{path}probability_within_radius_R_final.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), probmap[0].permute(1, 2, 0).cpu().numpy())
    epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
    # print(f"EPE: {epe.mean().cpu().item()}") #预测不可见像素的光流 不合理 误差很大
    # confidence_map = probmap
    if valid == None:
        valid = torch.ones(1, 1, H, W).bool()
    valid = valid.bool()
    gt_mask = valid[0].cpu().numpy()
    confidence_map_numpy = confidence_map.squeeze(1).cpu().numpy()
    confident_mask = (confidence_map_numpy > min_confidence).astype(bool)
    valid = gt_mask
    # valid = torch.ones(1, H, W).bool()
    epe = epe.cpu().numpy()[valid]
    px1 = (epe < 1.0).mean()
    px3 = (epe < 3.0).mean()
    px5 = (epe < 5.0).mean()
    epe = epe.mean()
    print(f"valid match EPE: {epe}, 1px: {100 * (1 - px1)}, 3px: {100 * (1 - px3)}, 5px: {100 * (1 - px5)}")

    # visualization of warped query image 
    fig, axis = plt.subplots(3, 3, figsize=(30, 30))
    estimated_flow_numpy = flow.squeeze().permute(1, 2, 0).cpu().numpy()
    gt_flow_numpy = flow_gt.squeeze().permute(1, 2, 0).cpu().numpy()
    
    warped_image2_numpy = remap_using_flow_fields(image2[0].permute(1, 2, 0).cpu().numpy(), estimated_flow_numpy[:, :, 0],
                                            estimated_flow_numpy[:, :, 1])
    gt_warped_image2_numpy = remap_using_flow_fields(image2[0].permute(1, 2, 0).cpu().numpy(), gt_flow_numpy[:, :, 0],
                                            gt_flow_numpy[:, :, 1])
    alpha = 0.5
    img_warped_overlay_on_target_masked = warped_image2_numpy * alpha + image1[0].permute(1, 2, 0).cpu().numpy() * alpha
    white_img = 255 * np.ones((H, W, 3))
    warped_image_certainty = confidence_map_numpy.reshape(H, W, 1) * warped_image2_numpy + (1 - confidence_map_numpy.reshape(H, W, 1)) * white_img
    cv2.imwrite(f"{path}warped_image_certainty.jpg", cv2.cvtColor(warped_image_certainty.astype(np.uint8), cv2.COLOR_RGB2BGR))
    warped_image_certainty = gt_mask.astype(float).reshape(H, W, 1) * gt_warped_image2_numpy + (1 - gt_mask.astype(float).reshape(H, W, 1)) * white_img
    cv2.imwrite(f"{path}ground_truth_warped_image_certainty.jpg", cv2.cvtColor(warped_image_certainty.astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}ground_truth_warped_image.jpg", cv2.cvtColor(gt_warped_image2_numpy.astype(np.uint8), cv2.COLOR_RGB2BGR))

    axis[0, 0].imshow(image2[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axis[0, 0].set_title('Query image')
    axis[0, 1].imshow(image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    axis[0, 1].set_title('Reference image')
    axis[1, 0].imshow(warped_image2_numpy.astype(np.uint8))
    axis[1, 0].set_title('Warped Query image according to estimated flow')
    axis[1, 1].imshow(img_warped_overlay_on_target_masked.astype(np.uint8))
    axis[1, 1].set_title('Warped query overlaid reference image')
    axis[2, 0].imshow(flow_to_image(estimated_flow_numpy))
    axis[2, 0].set_title('Estimated flow')
    axis[2, 1].imshow(flow_to_image(flow_gt[0].permute(1, 2, 0).cpu().numpy()))
    axis[2, 1].set_title('Ground truth flow')
    confident_warped = overlay_semantic_mask(warped_image2_numpy.astype(np.uint8), 
                                            ann=255 - confident_mask[0]*255, color=[255, 102, 51])
    axis[0, 2].imshow(confident_warped)
    axis[0, 2].set_title('Confident warped query \nimage according to estimated flow')
    axis[1, 2].imshow(confidence_map_numpy[0], vmin=0.0, vmax=1.0)
    axis[1, 2].set_title('Confident regions')
    img_warped_overlay_on_target_masked = warped_image2_numpy * alpha * \
        np.tile(np.expand_dims(confident_mask[0].astype(np.uint8), axis=2), (1, 1, 3)) \
                + (1 - alpha) * image1[0].permute(1, 2, 0).cpu().numpy()
    axis[2, 2].imshow(img_warped_overlay_on_target_masked.astype(np.uint8))
    axis[2, 2].set_title('Confident warped query overlaid reference image')
    fig.savefig(f"{path}flow_and_warp.jpg")
     


@torch.no_grad()
def demo_chairs(model, args, device=torch.device('cuda')):
    dataset = datasets.FlyingChairs(split='training')
    image1, image2, flow_gt, _ = dataset[1345]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('chairs', args, model, image1, image2, flow_gt)

def demo_sintel(model, args, device=torch.device('cuda')):
    dstype = 'final'
    aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
    # dataset = datasets.MpiSintel(split='training', dstype=dstype, aug_params=aug_params)
    dataset = datasets.MpiSintel(split='training', dstype=dstype)
    image1, image2, flow_gt, valid = dataset[350]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    valid = valid[None].to(device)
    valid = valid[None].to(device)
    demo_data('sintel', args, model, image1, image2, flow_gt, valid)

@torch.no_grad()
def demo_spring(model, args, device=torch.device('cuda'), split='train'):
    dataset = datasets.SpringFlowDataset(split=split)
    idx = 19198
    if split == 'train' or split == 'val':
        image1, image2, flow_gt, _ = dataset[idx]
    else:
        image1, image2,  _ = dataset[idx]
        h, w = image1.shape[1:]
        flow_gt = torch.zeros((2, h, w))

    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('spring', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_tartanair(model, args, device=torch.device('cuda')):
    dataset = datasets.TartanAir()
    image1, image2, flow_gt, _ = dataset[1070]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('tartanair', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_infinigen(model, args, device=torch.device('cuda')):
    dataset = datasets.Infinigen()
    image1, image2, flow_gt, _ = dataset[1000]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('infinigen', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_hd1k(model, args, device=torch.device('cuda')):
    dataset = datasets.HD1K()
    print(len(dataset))
    image1, image2, flow_gt, _ = dataset[0]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('hd1k', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_middlebury(model, args, device=torch.device('cuda')):
    dataset = datasets.Middlebury()
    image1, image2, flow_gt, _ = dataset[3]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('middlebury', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_dunhuang(model, args, device=torch.device('cuda')):
    # aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
    # dataset = datasets.dunhuang(split='training', dstype='val_homo+pertu_colorchange',aug_params=aug_params)
    dataset = datasets.dunhuang(split='training', dstype='val_homo+tps+pertu_colorchange512')
    image1, image2, flow_gt, valid = dataset[11]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    # valid = get_gt_correspondence_mask(flow_gt)
    valid = valid[None].to(device)
    valid = valid[None].to(device)
    print(valid.shape)
    demo_data('dunhuang', args, model, image1, image2, flow_gt, valid=valid)

@torch.no_grad()
def demo_Hpatches(model, args, device=torch.device('cuda')):
    im1_path = 'datasets/Hpatches/hpatches-sequences-release/v_azzola/3.ppm'
    im2_path = 'datasets/Hpatches/hpatches-sequences-release/v_azzola/4.ppm'

    image1 = torch.from_numpy(np.array(Image.open(im1_path), dtype=np.float32).transpose((2, 0, 1)))
    image2 = torch.from_numpy(np.array(Image.open(im2_path), dtype=np.float32).transpose((2, 0, 1)))

    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    try:
        flow_gt = torch.tensor(flow_gt, dtype=torch.float32).permute(2, 0, 1)
        flow_gt = flow_gt[None].to(device)
    except:
        flow_gt = torch.zeros([1, 2, H, W], device=device)
    valid = get_gt_correspondence_mask(flow_gt)
    valid = valid[None].to(device)
    demo_data('Hpatches', args, model, image1, image2, flow_gt, valid=valid)

@torch.no_grad()
def demo_MegaDepth(model, args, device=torch.device('cuda')):
    # aug_params = {'crop_size': args.image_size, 'min_scale': args.scale - 0.1, 'max_scale': args.scale + 1.0, 'do_flip': True}
    # dataset = datasets.dunhuang(split='training', dstype='val_homo+pertu_colorchange',aug_params=aug_params)
    mega = MegadepthBuilder(data_root="datasets/megadepth")
    
    # dataset = mega.build_scenes(
    #     split="test", min_overlap=0.01, ht=args.image_size[0], wt=args.image_size[1], 
    #     shake_t=32, return_data_dict = False
    # )    
    dataset = ConcatDataset(
        mega.build_scenes(split="test_loftr", ht=384, wt=512)
    )  # fixed resolution of 384,512    
    image1, image2, flow_gt, valid = dataset[5]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    # valid = get_gt_correspondence_mask(flow_gt)
    valid = valid[None][None].to(device)
    print(valid.shape)
    demo_data('megadepth-dense', args, model, image1, image2, flow_gt, valid=valid)


@torch.no_grad()
def demo_custom(model, args, device=torch.device('cuda')):
    flow_gt = None
    # image1 = cv2.imread('datasets/dunhuang/val_homo+pertu_colorchange512/images/image_0_img_2.jpg')
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # image2 = cv2.imread('datasets/dunhuang/val_homo+pertu_colorchange512/images/image_0_img_1.jpg')
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # flow_gt = frame_utils.read_gen("datasets/dunhuang/val_homo+pertu_colorchange512/flow/image_0_flow.flo")
    # image1 = cv2.imread('img1windows3.jpg')
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # image2 = cv2.imread('img2windows3.jpg')
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    # flow_gt = None
    # image1 = cv2.imread('datasets/dunhuang/specimiqRGB/021-1.png')
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # image2 = cv2.imread('datasets/dunhuang/specimiqRGB/021-2.png')
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
    # image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)

    im1_path = 'datasets/image_11_img_2.jpg'
    im2_path = 'datasets/image_11_img_1.jpg'

    image1 = torch.from_numpy(np.array(Image.open(im1_path), dtype=np.float32).transpose((2, 0, 1)))
    image2 = torch.from_numpy(np.array(Image.open(im2_path), dtype=np.float32).transpose((2, 0, 1)))

    H, W = image1.shape[1:]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    try:
        flow_gt = torch.tensor(flow_gt, dtype=torch.float32).permute(2, 0, 1)
        flow_gt = flow_gt[None].to(device)
    except:
        flow_gt = torch.zeros([1, 2, H, W], device=device)
    demo_data('custom_downsample', args, model, image1, image2, flow_gt)


@torch.no_grad()
def demo_dunhuang_case(model, args, case='synthetic_512_all', index=0, device=torch.device('cuda')):
    dataset = datasets.dunhuang(split='training', dstype=case)
    if len(dataset) == 0:
        raise ValueError(f"No image pairs found for case '{case}'. Check datasets/dunhuang/{case}/")
    if index < 0 or index >= len(dataset):
        raise IndexError(f"Index {index} is out of range for case '{case}' with {len(dataset)} pairs")

    image1, image2, flow_gt, valid = dataset[index]
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    valid = valid[None][None].to(device)

    demo_name = f"{case}_{index:04d}"
    demo_data(demo_name, args, model, image1, image2, flow_gt, valid=valid)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='model name', required=True, type=str)
    parser.add_argument('--restore_ckpt', help='model checkpoint path', required=True, type=str)
    parser.add_argument('--case', help='dunhuang case folder under datasets/dunhuang', default='synthetic_512_all', type=str)
    parser.add_argument('--index', help='image pair index inside the case folder', default=0, type=int)
    args = parse_args(parser)
    if args.model == 'RAFT':
        model = RAFT(args)
        load_ckpt(model, args, distributed=False)
    elif args.model == 'HomoRAFT':
        model = HomoRAFT(args)
        load_ckpt(model, args, distributed=False)
    elif args.model == 'GCSRAFT_certainty':
        model = GCSRAFT_certainty(args)
        load_ckpt(model, args, distributed=False)

    elif args.model == 'ORB':
        model = ORB(args)
    elif args.model == 'SIFT':
        model = SIFT(args)

    model = model.cuda()
    model.eval()

    demo_dunhuang_case(model, args, case=args.case, index=args.index)
    # if args.dataset == 'dunhuang':
    #     demo_dunhuang(model, args)

if __name__ == '__main__':
    main()
