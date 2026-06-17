import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import LayerNorm

class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * output_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * output_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.final = nn.Conv2d(dim, output_dim, kernel_size=1, padding=0)

    def forward(self, net, inp):
        x = torch.cat([net, inp], dim=1)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.final(input + x)
        return x

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class CertaintyHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(CertaintyHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv2(self.relu(self.conv1(x))))

# class BasicMotionEncoder(nn.Module):
#     def __init__(self, args):
#         super(BasicMotionEncoder, self).__init__()
#         cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
#         self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
#         self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
#         self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
#         self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
#         self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

#     def forward(self, flow, corr):
#         cor = F.relu(self.convc1(corr))
#         cor = F.relu(self.convc2(cor))
#         flo = F.relu(self.convf1(flow))
#         flo = F.relu(self.convf2(flo))

#         cor_flo = torch.cat([cor, flo], dim=1)
#         out = F.relu(self.conv(cor_flo))
#         return torch.cat([out, flow], dim=1)
    
class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        # self.convcer1 = nn.Conv2d(1, 128, 7, padding=3)
        # self.convcer2 = nn.Conv2d(128, 64, 3, padding=1)
        # self.finalconv = nn.Conv2d(64+64+192, 128-3, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)


    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        # cer = F.relu(self.convcer1(certainty))
        # cer = F.relu(self.convcer2(cer))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        # return torch.cat([out, flow], dim=1)
        return out

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        # self.refine = []
        # for i in range(args.num_blocks):
        #     if args.refine_blk == 'ConvNext':
        #         self.refine.append(ConvNextBlock(2*input_dim+hidden_dim, hidden_dim))
        #     elif args.refine_blk == "SepConvGRU":
        #         self.refine.append(SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim))
        # self.refine = nn.ModuleList(self.refine)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.certainty_head = CertaintyHead(input_dim=hidden_dim)

        # self.fusier = nn.Sequential(
        #     nn.Conv2d(128+128-2, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 128+128-2, 3, padding=1))
        
        # self.fusier = torch.nn.ModuleList([ConvBlock(128+128-2, 128+128-2, kernel_size=3, stride=1, padding=1)
        #                                         for i in range(args.num_blocks)])
                
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, certainty, upsample=True):
        motion_features = self.encoder(flow, corr)
        # motion_features = self.encoder(flow, corr, certainty)
        inp = torch.cat([inp, motion_features], dim=1)
        # inp = inp + self.fusier(torch.cat([inp, certainty], dim=1))
        # for conv in self.fusier:
        #     inp = inp + conv(inp)
        # inp = inp + self.fusier(inp)
        # inp = inp * certainty
        inp = torch.cat([inp, flow], dim=1)
        # for blk in self.refine:
        #     net = blk(net, inp)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        certainty = self.certainty_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow, certainty



