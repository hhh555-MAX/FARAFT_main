import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myupdate import BasicUpdateBlock
# from update_raft import BasicUpdateBlock

from extractor_raft import BasicEncoder
from corr_raft import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, InputPadder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT_certainty(nn.Module):
    def __init__(self, args):
        super(RAFT_certainty, self).__init__()
        self.args = args

        
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2*torch.ones(1))
        # self.hidden_init = nn.Sequential(
        #     nn.Conv2d(1+2+128, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 1, padding=0)
        # )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    
    def global_correlation_softmax(self, feature0, feature1, corr_pyramid):
        # global correlation
        b, c, h, w = feature0.shape
        _, _, h2, w2 = feature1.shape
        corr = corr_pyramid[0]
        correlation = corr.reshape(b, h, w, h, w)
        # flow from softmax
        init_grid = coords_grid(b, h, w, device=correlation.device)  # [B, 2, H, W]
        grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

        # prob = F.softmax(correlation, dim=1) * F.softmax(correlation, dim=2)  # [B, H*W, H*W]
        prob = F.softmax(correlation, dim=2)  # [B, H*W, H*W]
        correspondence = torch.matmul(prob, grid)  # [B, H*W, 2]
        # initial certainty from variance
        # variance = x-axis variance + y-axis variance
        variance = torch.matmul(prob, (grid - correspondence) ** 2).sum(dim=2).view(b, h*w) # [B, H*W]
        # var_max, _ = variance.max(dim=1)
        # variance = variance / var_max.view(b, 1) # [B, H*W]
        # certainty = 1 - variance # [B, H*W]
        certainty = self.beta + self.gamma * variance
        certainty = torch.sigmoid(certainty)
        certainty = certainty.view(b, 1, h, w) #[B, 1, H, W]

        correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        # # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
        # flow = correspondence - init_grid

        # return flow, prob
        return correspondence, certainty

    def upsample_flow(self, flow_certainty, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = flow_certainty.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        if C == 2:
            up_flow = F.unfold(8 * flow_certainty, [3,3], padding=1)
        elif C == 1: #certainty
            up_flow = F.unfold(flow_certainty, [3,3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, flow_gt=None, test_mode=False):
        """ Estimate optical flow between pair of frames """
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters
        if flow_gt is None:
            flow_gt = torch.zeros(N, 2, H, W, device=image1.device)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        N, _, H, W = image1.shape

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)


        coords0, coords1 = self.initialize_flow(fmap1)

        # # Correlation as initialization
        # N, fC, fH, fW = fmap1.shape
        # corrMap = corr_fn.corrMap

        # #_, coords_index = torch.max(corrMap, dim=-1) # no gradient here
        # softCorrMap = F.softmax(corrMap, dim=2) * F.softmax(corrMap, dim=1) # (N, fH*fW, fH*fW)

        # if flow_init is not None:
        #     coords1 = coords1 + flow_init
        # else:
        #     coords1, certainty = self.global_correlation_softmax(fmap1, fmap2, corr_fn.corr_pyramid) #TODO 把CorrBlock里面的cv放进去，不要重复计算
            # coords1 = self.pos_embed(corr_fn.corr_pyramid[0])

            
            # print('matching as init')
            # # mutual match selection
            # match12, match_idx12 = softCorrMap.max(dim=2) # (N, fH*fW)
            # match21, match_idx21 = softCorrMap.max(dim=1)

            # for b_idx in range(N):
            #     match21_b = match21[b_idx,:]
            #     match_idx12_b = match_idx12[b_idx,:]
            #     match21[b_idx,:] = match21_b[match_idx12_b]

            # matched = (match12 - match21) == 0  # (N, fH*fW)
            # coords_index = torch.arange(fH*fW).unsqueeze(0).repeat(N,1).to(softCorrMap.device)
            # coords_index[matched] = match_idx12[matched]

            # # matched coords
            # coords_index = coords_index.reshape(N, fH, fW)
            # coords_x = coords_index % fW
            # coords_y = coords_index // fW

            # coords_xy = torch.stack([coords_x, coords_y], dim=1).float()
            # coords1 = coords_xy            

        flow_predictions = []
        certainty_predictions = []
        flow = coords1 - coords0
        flow_up = upflow8(flow)
        B, C, H, W = fmap1.shape
        certainty = torch.ones(size=(B, 1, H, W)).cuda() #TODO 替换成从cv里计算的方式
        flow_predictions.append(flow_up)
        certainty_up = upflow8(certainty)/8.0
        certainty_predictions.append(certainty_up)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            # net = self.hidden_init(torch.cat((certainty, flow, net), dim=1))
            net = torch.tanh(net)
            inp = torch.relu(inp)


        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, certainty = self.update_block(net, inp, corr, flow, certainty)
                # net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
                certainty_up = upflow8(certainty)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                certainty_up = self.upsample_flow(certainty, up_mask)
            
            flow_predictions.append(flow_up)
            certainty_predictions.append(certainty_up)

        for i in range(len(flow_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])
            certainty_predictions[i] = padder.unpad(certainty_predictions[i])
        return {'final': flow_predictions[-1], 'flow': flow_predictions, 'certainty': certainty_predictions}
