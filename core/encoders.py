import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from extractor_raft import BasicEncoder

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	  return self.layer(x)


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, 
                 early_exit = False, amp = False, amp_dtype = torch.float16, if_pyramid=False) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.if_pyramid = if_pyramid

    def forward(self, x, **kwargs):
        net = self.net
        feats = {1:x}
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[2] = x 
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[4] = x 
        x = net.layer2(x)
        feats[8] = x
        if self.early_exit:
            return feats
        x = net.layer3(x)
        feats[16] = x
        x = net.layer4(x)
        feats[32] = x
        if self.if_pyramid is True:
            return feats
        else:
            return feats[8]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16, if_pyramid=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.if_pyramid = if_pyramid

    def forward(self, x, **kwargs):
        feats = {}
        scale = 1
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                feats[scale] = x
                scale = scale*2
            x = layer(x)
        if self.if_pyramid is True:
            return feats
        else:
            return feats[8]
    

class Encoder(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, cnn_backbone = False, dinov2_weights = None, 
                 amp_dtype = torch.float16, if_pyramid=False):
        super().__init__()

        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from transformer import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        self.if_pyramid = if_pyramid
        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        cnn_kwargs['if_pyramid'] = True
        if cnn_backbone == 'resnet50':
            self.cnn = ResNet50(**cnn_kwargs)
        elif cnn_backbone == 'vgg19':
            self.cnn = VGG19(**cnn_kwargs)
        # elif cnn_backbone == 'basicencoder':
        #     self.cnn = BasicEncoder(**cnn_kwargs)
        else:
            raise NotImplementedError
        self.amp = amp
        # self.amp_dtype = amp_dtype
        # if self.amp:
        #     dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
        self.zip_layer1 = [BasicLayer(1024, 256, 1, padding=0)]
        self.zip_layer2 = [BasicLayer(512, 256, 1, padding=0)]
        self.fusion = [nn.Sequential(
            BasicLayer(512, 256, 3, stride=1),
            BasicLayer(256, 256, 3, stride=1),
            nn.Conv2d (256, 256, 1, padding=0)
        )]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)    

                    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)
        coarse_size = 560
        with torch.no_grad():
            x = F.interpolate(x, (coarse_size), mode='bilinear')
            if self.dinov2_vitl14[0].device != x.device:
                # self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device)
                self.zip_layer1[0] = self.zip_layer1[0].to(x.device)
                self.zip_layer2[0] = self.zip_layer2[0].to(x.device)
                self.fusion[0] = self.fusion[0].to(x.device)
            # dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
            dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x)
            features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(
                B,1024,coarse_size//14, coarse_size//14)
            del dinov2_features_16
            features_16 = self.zip_layer1[0](features_16)
            feature_pyramid[8] = self.zip_layer2[0](feature_pyramid[8])
        features_16_up = F.interpolate(features_16, (H//8, W//8), mode='bilinear')
        feature_pyramid[8] = self.fusion[0](torch.cat([features_16_up, feature_pyramid[8]], dim=1))
        
        if self.if_pyramid is True:
            return feature_pyramid
        else:
            return feature_pyramid[8]
    
