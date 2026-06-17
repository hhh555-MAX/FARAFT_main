import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    certainty_loss = 0.0
    nf_predictions = []
    ce_predictions = []
    for i in range(n_predictions):
        i_loss = (output['flow'][i] - flow_gt).abs()
        nf_predictions.append(i_loss)
        if 'certainty' in output:
            ce_loss =  F.binary_cross_entropy_with_logits(output['certainty'][i].squeeze(1), valid)
            ce_predictions.append(ce_loss)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # valid = (flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000)
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = nf_predictions[i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
        if 'certainty' in output:
            certainty_loss += i_weight*ce_predictions[i].mean()

    return {'loss': flow_loss + ce_weight * certainty_loss, 'flow_loss':flow_loss, 'certainty_loss':certainty_loss}


# # exclude extremly large displacements
# MAX_FLOW = 400
# SUM_FREQ = 100
# VAL_FREQ = 5000

def searaft_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

    return {'loss': flow_loss, 'flow_loss':flow_loss}

# GLUNet
def glunet_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    certainty_loss = 0.0
    nf_predictions = []
    # weights_level_loss = [0.32, 0.08, 0.02, 0.01]
    weights_level_loss = [1]
    assert len(weights_level_loss) == n_predictions

    for i in range(n_predictions):
        _, _, h, w = output['flow'][i].shape # 16x16 32x32 1/8 1/4
        flow_scaled = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=False)
        i_loss = (output['flow'][i] - flow_scaled).norm(dim=1)
        nf_predictions.append(i_loss)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # valid = (flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000)
    valid = (valid >= 0.5) & (mag < max_flow)
    
    for i in range(n_predictions):
        i_weight = weights_level_loss[i]
        # i_weight = gamma ** (n_predictions - i - 1)
        loss_i = nf_predictions[i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach()))
        # flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
        flow_loss += i_weight * ((final_mask * loss_i).sum())

    return {'loss': flow_loss + ce_weight * certainty_loss, 'flow_loss':flow_loss, 'certainty_loss':certainty_loss}

def dkm_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    certainty_loss = 0.0
    nf_predictions = []
    ce_predictions = []
    for i in range(n_predictions):
        _, _, h, w = output['flow'][i].shape # 1/32 1/16 1/8 1/4 1/2 1 训练和推理固定分辨率为512*512
        flow_scaled = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=False)
        i_loss = (output['flow'][i] - flow_scaled).norm(dim=1)
        nf_predictions.append(i_loss)
        if 'certainty' in output:
            valid_scaled = F.interpolate(valid.unsqueeze(1), (h, w), mode='bilinear', align_corners=False).squeeze(1)
            ce_loss =  F.binary_cross_entropy_with_logits(output['certainty'][i].squeeze(1), valid_scaled)
            ce_predictions.append(ce_loss)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # valid = (flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000)
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = nf_predictions[i]
        _, h, w = loss_i.shape
        valid_scaled = F.interpolate(valid.unsqueeze(1).float(), (h, w), mode='bilinear', align_corners=False).squeeze(1).bool()
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid_scaled[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
        if 'certainty' in output:
            certainty_loss += i_weight*ce_predictions[i].mean()

    return {'loss': flow_loss + ce_weight * certainty_loss, 'flow_loss':flow_loss, 'certainty_loss':certainty_loss}

def homoraft_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output['flow'])
    flow_loss = 0.0
    certainty_loss = 0.0
    corner_loss = 0.0
    nf_predictions = []
    four_point_predictions = []
    for i in range(n_predictions):
        i_loss = (output['flow'][i] - flow_gt).abs()
        nf_predictions.append(i_loss)
        N, _, H, W = flow_gt.shape
        four_point_gt = torch.zeros((N, 2, 2, 2)).to(flow_gt.device)
        four_point_gt[:, 0, 0] = torch.Tensor(flow_gt[:, :, 0, 0])
        four_point_gt[:, 0, 1] = torch.Tensor(flow_gt[:, :, 0, -1])
        four_point_gt[:, 1, 0] = torch.Tensor(flow_gt[:, :, -1, 0])
        four_point_gt[:, 1, 1] = torch.Tensor(flow_gt[:, :, -1, -1])    # exlude invalid pixels and extremely large diplacements
        four_point_gt[:, :, :, 0] /= W-1
        four_point_gt[:, :, :, 1] /= H-1
        four_point_loss = torch.abs(output['four_point'][i] - four_point_gt).sum(dim=-1).mean(-1).mean(-1)
        four_point_predictions.append(four_point_loss)
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # valid = (flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000)
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = nf_predictions[i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
        corner_loss += i_weight*four_point_predictions[i].mean()
    return {'loss': flow_loss + ce_weight * corner_loss, 'flow_loss':flow_loss, 'corner_loss':corner_loss}
