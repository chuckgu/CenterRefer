import torch.nn as nn
import torch.nn.functional as F
import torch
from modeling.encoder_fusion import *
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class CenterRefer(nn.Module):
    def __init__(self,
                 vis_emb_net="",sync_bn=True):
        super().__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d


        self.vis_emb_net = vis_emb_net
        self.encoder_fusion=Encoder_fusion()
        # self.center_attention=CGSA()
        self.mask_conv = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, 1, kernel_size=1, stride=1))

        self.heatmap_conv = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(128, 1, kernel_size=1, stride=1))

    def forward(self, input):

        (img, text, heatmap_gt) = input
        low_level_feat, low_level_feat_8, x = self.vis_emb_net.forward_encoder(img)  # [b,256,80,80], [b,512,40,40], [b,256,20,20]

        fusion_feat=self.encoder_fusion((low_level_feat, low_level_feat_8, x,text))
        # att_fusion_feat=self.center_attention(fusion_feat)

        center_heatmap = self.heatmap_conv(fusion_feat)
        x = self.mask_conv(fusion_feat)
        # x = self.vis_emb_net.decoder(x, fusion_feat) # x.shape=[b,80,129,129] [b,80,80,80]

        x = F.interpolate(x, size=img.size()[2:], mode='bilinear', align_corners=True)  # x.shape=[b,80,513,513] [b,80,320,320]
        # center_heatmap = F.interpolate(center_mask, size=img.size()[2:], mode='bilinear', align_corners=True)

        return x, center_heatmap

    def get_params(self):
        modules = [self.encoder_fusion,self.mask_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                # if self.freeze_bn:
                #     if isinstance(m[1], nn.Conv2d):
                #         for p in m[1].parameters():
                #             if p.requires_grad:
                #                 yield p
                # else:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_params_slow(self):
        modules = [self.heatmap_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                # if self.freeze_bn:
                #     if isinstance(m[1], nn.Conv2d):
                #         for p in m[1].parameters():
                #             if p.requires_grad:
                #                 yield p
                # else:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
