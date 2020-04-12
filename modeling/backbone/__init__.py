from modeling.backbone import resnet


def build_backbone(backbone,
    output_stride, BatchNorm, pretrained=False, imagenet_pretrained_path=""
):
    return resnet.ResNet101(
        output_stride,
        BatchNorm,
        pretrained=pretrained
    )
