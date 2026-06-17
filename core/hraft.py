import torch
import torch.nn as nn
import torch.nn.functional as F

from ast import iter_child_nodes
from mcnet.extractor import *
from mcnet.update import *
from mcnet.corr import *
from mcnet.utils import *
from mcnet.flow_utils import *
from mcnet.homo_utils import *
from update_raft import BasicUpdateBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8, InputPadder
from extractor_raft import HomoRAFTEncoder
from corr_raft import CorrBlock, AlternateCorrBlock

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


class HomoRAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.fnet = BasicEncoder(output_dim=96, norm_fn='instance')
        self.update_blocks = nn.ModuleList([CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=4),
                                    CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=5),
                                    CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=6)])
                
        self.downsample = [16, 8]
        self.iter = [2, 2]
        # args.corr_levels = 1
        args.corr_levels = 4

        # self.update_blocks = nn.ModuleList([CorrelationDecoder(args=args, input_dim=81 * args.corr_levels, hidden_dim=64, output_dim=2, downsample=5)])
        # self.update_blocks = nn.ModuleList([CorrelationDecoder(args=args, input_dim=81, hidden_dim=64, output_dim=2, downsample=5)])
        # self.downsample = [1]
        # self.iter = [4]
        self.memory = {"deltaD":[], "scale":[], "delta_ace":[], "iteration":[]}
        self.instance_experts = []
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_radius = 4
        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.fnet = HomoRAFTEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = HomoRAFTEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

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

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
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
        device = image1.device

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):

            fmap1 = self.fnet(image1)
            fmap2 = self.fnet(image2)
        # fmap1 = fmap1.float()
        # fmap2 = fmap2.float()
        
        four_point_disp = torch.zeros((N, 2, 2, 2)).to(image1.device)
        four_point_predictions = []

        for idx in range(len(self.downsample)):
            downsample = self.downsample[idx]
            # idx = self.downsample.index(downsample)
            corr_fn = LocalCorr(fmap1[idx], fmap2[idx]) # 通道数81
            # fmap1_64 = torch.nn.functional.interpolate(input=fmap1[idx], size=(64, 64), mode='bilinear',
            #                                         align_corners=False)
            # fmap2_64 = torch.nn.functional.interpolate(input=fmap2[idx], size=(64, 64), mode='bilinear',
            #                                         align_corners=False)
            # corr_fn = LocalCorr(fmap1_64, fmap2_64) # 通道数81
            coords0, coords1 = self.initialize_flow(fmap1[idx])

            for _ in range(self.iter[idx]):
                coords1 = disp_to_coords(four_point_disp, coords0, downsample=downsample)              
                corr = corr_fn(coords1)   
                four_point_delta = self.update_blocks[idx](corr)
                four_point_disp =  four_point_disp + four_point_delta
                four_point_reshape = four_point_disp.permute(0,2,3,1).reshape(-1,4,2) # [top_left, top_right, bottom_left, bottom_right], [-1, 4, 2]
                
                four_point_predictions.append(four_point_reshape)
        coords1 = disp_to_coords(four_point_disp, coords0, downsample=downsample)
        flow_64 = coords1 - coords0

        # coarse_flow = torch.nn.functional.interpolate(input=flow_64, size=(H // 8, W // 8), mode='bilinear',
        #                                            align_corners=False)
        # coarse_flow[:, 0, :, :] *= (1.0 *W / (128 * 8.0))
        # coarse_flow[:, 1, :, :] *= (1.0 *H / (128 * 8.0))

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1[-1], fmap2[-1], radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1[-1], fmap2[-1], radius=self.args.corr_radius) #通道数4*81
            # corr_fn = LocalCorr(fmap1[idx], fmap2[idx], self.args.corr_levels)
            # corr_fn = LocalCorr(fmap1[idx], fmap2[idx])
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet[-1], [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        coords0, coords1 = self.initialize_flow(fmap1[-1])

        coords1 = coords1 + flow_64
        flow = coords1 - coords0
        flow_up = upflow8(flow)
        flow_predictions = []
        flow_predictions.append(flow_up)
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        for i in range(len(flow_predictions)):
            flow_predictions[i] = padder.unpad(flow_predictions[i])

        return {'final': flow_predictions[-1], 'flow': flow_predictions}
