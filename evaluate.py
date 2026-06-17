import sys
sys.path.append('core')
import argparse
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import cv2
from glob import glob
import os.path as osp

from config.parser import parse_args

import datasets
from raft import RAFT
from gcsraft import GCSRAFT
from gcsraft_certainty import GCSRAFT_certainty
from raft_certainty import RAFT_certainty
# from faraft import FARAFT, GLUNet
from hraft import HRAFT, mcnet
from orb_sift import ORB, SIFT

from utils.utils import resize_data, load_ckpt
from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask, unormalise_and_convert_mapping_to_flow
from utils_data.geometric_transformation_sampling.aff_homo_tps_generation import HomographyGridGen
from datasets import MegadepthBuilder
from utils.utils import warp_kpts
from torch.utils.data import ConcatDataset

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    # flow_final = output['flow'][-1]
    flow_final = output['final']
    if 'info' in output and output['info'] is not None:
        info_final = output['info'][-1]
    else:
        info_final = None
    # confidence_map = output['uncertainty']['inv_cyclic_consistency_error']
    # return flow_final, info_final, confidence_map
    return flow_final, info_final
def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    if info is not None:
        info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    else:
        info_down = None
    return flow_down, info_down

def get_probability_map(info, args, R=1):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    # TODO PDCNet+里有公式（7） 找到两篇论文里log_b和方差var_map的关系 注意是无穷范数
    p_r = torch.sum(weight * (1 - torch.exp(- R/torch.exp(log_b)))**2, dim=1, keepdim=True) 
    return p_r

@torch.no_grad()
def validate_chairs(args, model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1, 
        pin_memory=False, shuffle=False, num_workers=2, drop_last=False)
    loop = tqdm((val_loader), total=len(val_loader))
    for i_batch, data_blob in enumerate(loop):
        image1, image2, flow_gt, _ = [x.cuda(non_blocking=True) for x in data_blob]

        flow, info = calc_flow(args, model, image1, image2)

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).cpu().numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairsEPE': epe}

@torch.no_grad()
def validate_dunhuang(args, model, dstype_list=['val_homo_nocolorchange', 'val_homo+pertu_nocolorchange', 'val_homo_colorchange', 'val_homo+pertu_colorchange']):
    """ Peform validation using the dunhuang(kari) split """
    # for dstype in ['generate_dataset_hard', 'generate_dataset_homo', 'generate_dataset_tangphoto_val','generate_dataset_tangphoto_val_colortrans']:
    result = {}
    for dstype in dstype_list:
        val_dataset = datasets.dunhuang(split='training', dstype=dstype)
        val_loader = data.DataLoader(val_dataset, batch_size=1, 
            pin_memory=False, shuffle=False, num_workers=1, drop_last=False)
        epe_list = np.array([], dtype=np.float32)
        re_list = np.array([], dtype=np.float32)
        px1_list = np.array([], dtype=np.float32)
        px3_list = np.array([], dtype=np.float32)
        px5_list = np.array([], dtype=np.float32)
        print(f"load data success {len(val_loader)}")
        loop = tqdm((val_loader), total=len(val_loader))
        for i_batch, data_blob in enumerate(loop):
        # for i_batch, data_blob in enumerate(val_loader):
            image1, image2, flow_gt, _ = [x.cuda(non_blocking=True) for x in data_blob]
            # flow, info = calc_flow(args, model, image1, image2) # calc函数里的降采样升采样虽然加快了推理时间，但大大降低精度
            flow, info = forward_flow(args, model, image1, image2)
            # flow, info, confidence_map = forward_flow(args, model, image1, image2)

            # probmap = get_probability_map(info, args)
            # confidence_map = probmap # TODO raft预测不确定度的效果差，不如循环一致性损失，但也不一定，可能是概率阈值太低了？
            # warped_image2 = warp(image2, flow)
            # re = torch.sum((image1 - warped_image2).abs(), dim=1)
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt() #B, H, W
            # px1 = (epe < 1.0).float().mean(dim=[1, 2]).cpu().numpy()
            # px3 = (epe < 3.0).float().mean(dim=[1, 2]).cpu().numpy()
            # px5 = (epe < 5.0).float().mean(dim=[1, 2]).cpu().numpy()
            # epe = epe.mean(dim=[1, 2]).cpu().numpy()
            
            # confidence_map_numpy = confidence_map.squeeze(1).cpu().numpy()
            gt_mask = get_gt_correspondence_mask(flow_gt).cpu().numpy()
            # confident_mask = (confidence_map_numpy > 0.30).astype(bool)
            valid = gt_mask

            epe = epe.cpu().numpy()[valid]
            # re = re.cpu().numpy()[valid]
            px1 = (epe < 1.0).mean()
            px3 = (epe < 3.0).mean()
            px5 = (epe < 5.0).mean()
            epe = epe.mean()
            # re = re.mean()
            epe_list = np.append(epe_list, epe)
            # re_list = np.append(re_list, re)
            px1_list = np.append(px1_list, px1)
            px3_list = np.append(px3_list, px3)
            px5_list = np.append(px5_list, px5)

        # re = np.mean(re_list)
        epe = np.mean(epe_list)
        px1 = np.mean(px1_list)
        px3 = np.mean(px3_list)
        px5 = np.mean(px5_list)
        print(f"Validation {dstype} EPE: {epe:.3f}, 1px: {100 * (1 - px1):.3f}, 3px: {100 * (1 - px3):.3f}, 5px: {100 * (1 - px5):.3f}")
        result[dstype] = {'EPE':epe, '1px':100 * (1 - px1), '3px': 100 * (1 - px3), '5px': 100 * (1 - px5),
                          'epe_list': epe_list}
    return result



def eval(args):
    args.gpus = [0]
    if args.model == 'RAFT':
        model = RAFT(args)
        load_ckpt(model, args, distributed=False)
    elif args.model == 'GCSRAFT_certainty':
        model = GCSRAFT_certainty(args)
        load_ckpt(model, args, distributed=False)
    elif args.model == 'ORB':
        model = ORB(args)
    elif args.model == 'SIFT':
        model = SIFT(args)
    else:
        raise NotImplementedError
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        if args.dataset == 'dunhuang':
            # if args.model == 'ORB' or args.model == 'SIFT':
            #     result = validate_dunhuang_cv2(args, model,dstype_list=['val_homo_nocolorchange', 'val_homo+pertu_nocolorchange', 
            #             'val_homo_colorchange', 'val_homo+pertu_colorchange','val_homo+tps+pertu_colorchange'])
            # else: 
            result = validate_dunhuang(args, model, dstype_list=['val_homo_nocolorchange', 'val_homo+pertu_nocolorchange', 
            'val_homo_colorchange', 'val_homo+pertu_colorchange','val_homo+tps+pertu_colorchange'])
            # result = validate_dunhuang(args, model, dstype_list=['val_homo_colorchange'])
            for dstype, value in result.items():
                with open(os.path.join('demo', dstype+'EPErecords.txt'), 'w') as file:
                    i = 0
                    for imgepe in value['epe_list']:
                        file.write(str(i) + '\t' + str(imgepe) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--model', help='model name', required=True, type=str)
    parser.add_argument('--restore_ckpt', help='model checkpoint path', type=str)
    args = parse_args(parser)
    eval(args)

if __name__ == '__main__':
    main()

