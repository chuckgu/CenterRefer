import torch.nn as nn
import torch.nn.functional as F
import torch

from modeling.aspp import build_aspp
from modeling.backbone import build_backbone
from modeling.decoder import build_decoder
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLab(nn.Module):
    def __init__(
        self,
        output_stride=16,
        num_classes=21,
        sync_bn=True,
        freeze_bn=False,
        pretrained=True,
        global_avg_pool_bn=True,
        pretrained_path="",
    ):
        super().__init__()
        backbone = 'resnet'
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        # self.backbone = build_backbone(
        #     output_stride,
        #     BatchNorm,
        #     pretrained=False,
        # )
        # self.aspp = build_aspp(output_stride, BatchNorm, global_avg_pool_bn)
        # self.decoder = build_decoder(num_classes, BatchNorm)

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

        if pretrained:
            self._load_pretrained_model(pretrained_path)

    def _load_pretrained_model(self, pretrained_path):
        """
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        """

        pretrain_dict = torch.load(pretrained_path)["state_dict"]
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():

            k = k[:-1]
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)

        self.load_state_dict(state_dict)


    def forward(self, input):#[b,3,320,320]
        x, low_level_feat = self.backbone(input) # x.shape=[b,2048,33,33] [b,2048,20,20]
        x = self.aspp(x)  # x.shape=[b,256,33,33] [b,256,20,20]
        x = self.decoder(x, low_level_feat) # x.shape=[b,80,129,129] [b,80,80,80]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # x.shape=[b,80,513,513] [b,80,320,320]

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p