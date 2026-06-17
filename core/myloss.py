
# from utils_flow.flow_and_mapping_operations import get_gt_correspondence_mask
# # exclude extremly large displacements
# MAX_FLOW = 400
# SUM_FREQ = 100
# VAL_FREQ = 5000

# def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
#     """ Loss function defined over sequence of flow predictions """
#     n_predictions = len(output['flow'])
#     flow_loss = 0.0
#     certainty_loss = 0.0
#     nf_predictions = []
#     ce_predictions = []
#     for i in range(n_predictions):
#         i_loss = (output['flow'][i] - flow_gt).abs()
#         nf_predictions.append(i_loss)
#         if 'certainty' in output:
#             ce_loss =  F.binary_cross_entropy_with_logits(output['certainty'][i].squeeze(1), valid)
#             ce_predictions.append(ce_loss)
#     # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()

#     # valid = (flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000)
#     valid = (valid >= 0.5) & (mag < max_flow)
#     for i in range(n_predictions):
#         i_weight = gamma ** (n_predictions - i - 1)
#         loss_i = nf_predictions[i]
#         final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
#         flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())
#         if 'certainty' in output:
#             certainty_loss += i_weight*ce_predictions[i].mean()

#     return {'loss': flow_loss + ce_weight * certainty_loss, 'flow_loss':flow_loss, 'certainty_loss':certainty_loss}

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW, ce_weight=1):
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
