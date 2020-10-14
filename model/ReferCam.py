import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align

from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import random
from collections import OrderedDict
from .darknet import *

from utils.parsing_metrics import *
from utils.utils import *
from train_yolo import get_args


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def bbox_transform_inv(boxes, deltas, batch_size,grid_size):
    boxes=boxes/grid_size

    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    # dw = deltas[:, :, 2::4]
    # dh = deltas[:, :, 3::4]

    # pred_ctr_x = F.sigmoid(dx) * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    # pred_ctr_y = F.sigmoid(dy) * heights.unsqueeze(2) + ctr_y.unsqueeze(2)

    pred_ctr_x = ctr_x.unsqueeze(2)
    pred_ctr_y = ctr_y.unsqueeze(2)

    coeff = random.uniform(-0.1, 0.1) + 1.0
    pred_w = coeff * widths.unsqueeze(2)#torch.exp(dw) * widths.unsqueeze(2)
    coeff = random.uniform(-0.1, 0.1) + 1.0
    pred_h = coeff * heights.unsqueeze(2)#torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = boxes.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes*grid_size

def bbox_sample_inv(boxes,size):


    widths = boxes[2] - boxes[0] + 1.0
    heights = boxes[3] - boxes[1] + 1.0
    ctr_x = boxes[0] + 0.5 * widths
    ctr_y = boxes[1] + 0.5 * heights

    coeff=np.random.normal(0,0.05) + 1.0

    pred_w=coeff*widths

    coeff=np.random.normal(0,0.05) + 1.0

    pred_h =coeff*heights

    pred_boxes = boxes.clone()
    # x1
    pred_boxes[0::4] = ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[1::4] = ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[2::4] = ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[3::4] = ctr_y + 0.5 * pred_h

    return pred_boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape-1)
        boxes[i,:,1::4].clamp_(0, im_shape-1)
        boxes[i,:,2::4].clamp_(0, im_shape-1)
        boxes[i,:,3::4].clamp_(0, im_shape-1)

    return boxes

def more_boxes(boxes, num, size):
    batch_size=boxes.size(0)
    boxes_more=boxes.new(batch_size, num, 4).zero_()

    for i in range(batch_size):
        gt_box=boxes[i,0]
        boxes_more[i,0]=gt_box
        for j in range(1,num):
            boxes_more[i, j]=bbox_sample_inv(gt_box,size)

    boxes_more = clip_boxes(boxes_more, size, batch_size)
    return boxes_more


def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps


class Interpolate(nn.Module):
    def __init__(self, size, mode= 'bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners = True)
        return x

class ReferCam(nn.Module):
    def __init__(self, leaky=False):
        super(ReferCam, self).__init__()

        embin_size=512+3#512+3#512*2+8
        emb_size=256
        cam_size=40

        # self.compress_lang = torch.nn.Sequential(
        #   nn.Linear(emb_size*2, emb_size),
        #   nn.BatchNorm1d(emb_size),
        #   nn.ReLU(),
        # )

        self.fcn_emb=torch.nn.Sequential(
                # ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                # ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(embin_size, emb_size, 3, 1, 1, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                # Interpolate(size=(cam_size, cam_size), mode='bilinear'),
        )
        # self.mapping_visu = nn.Sequential(OrderedDict([
        #     ('0', ConvBatchNormReLU(1024, emb_size, 1, 1, 0, 1, leaky=leaky)),
        #     ('1', ConvBatchNormReLU(512, emb_size, 1, 1, 0, 1, leaky=leaky)),
        #     ('2', ConvBatchNormReLU(256, emb_size, 1, 1, 0, 1, leaky=leaky))
        # ]))

        # self.fcn_emb = nn.Sequential(OrderedDict([
        #     ('0', torch.nn.Sequential(
        #         ConvBatchNormReLU(embin_size, emb_size, 3, 1, 1, 1, leaky=leaky),
        #         # ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         # Interpolate(size=(cam_size, cam_size), mode='bilinear'),
        #     )),
        #     ('1', torch.nn.Sequential(
        #         ConvBatchNormReLU(embin_size, emb_size, 3, 1, 1, 1, leaky=leaky),
        #         # ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         # Interpolate(size=(cam_size, cam_size), mode='bilinear'),
        #     )),
        #     ('2', torch.nn.Sequential(
        #         ConvBatchNormReLU(embin_size, emb_size, 3, 1, 1, 1, leaky=leaky),
        #         # ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         ConvBatchNormReLU(emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         # Interpolate(size=(cam_size, cam_size), mode='bilinear'),
        #     )),
        # ]))

        self.fcn_out=torch.nn.Sequential(
                ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(emb_size // 2, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size // 2, 80, kernel_size=1),
        )
        # self.fcn_out = nn.Sequential(OrderedDict([
        #     ('0', torch.nn.Sequential(
        #         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
        #         nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        #     ('1', torch.nn.Sequential(
        #         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
        #         nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        #     ('2', torch.nn.Sequential(
        #         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
        #         nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        # ]))

        seg_emb_size = emb_size + 3
        self.refine = nn.Sequential(OrderedDict([
            ('0', torch.nn.Sequential(
                ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky), )),
            ('1', torch.nn.Sequential(
                ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky), )),
            ('2', torch.nn.Sequential(
                ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                ConvBatchNormReLU(seg_emb_size, emb_size, 1, 1, 0, 1, leaky=leaky), )),
        ]))


        # self.fcn_emb = nn.Sequential(OrderedDict([
        #     ('0', torch.nn.Sequential(
        #         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         # ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         Interpolate(size=(cam_size, cam_size), mode='bilinear'),)),
        #     ('1', torch.nn.Sequential(
        #         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         # ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         Interpolate(size=(cam_size, cam_size), mode='bilinear'), )),
        #     ('2', torch.nn.Sequential(
        #         ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         # ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),
        #         Interpolate(size=(cam_size, cam_size), mode='bilinear'), )),
        # ]))
        #
        # self.fcn_out = nn.Sequential(OrderedDict([
        #     ('0', torch.nn.Sequential(
        #         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
        #         nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        #     ('1', torch.nn.Sequential(
        #         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
        #         nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        #     ('2', torch.nn.Sequential(
        #         ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
        #         nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        # ]))

        # self.avg_pool =  nn.AvgPool2d(20)

    def forward(self, input):
        # args = get_args()

        # if self.training:
        #     (intmd_fea, image, flang, bbox, pred_anchor, args)  =input
        #     anchors_full = get_archors_full(args)
        #     batch_size=args.batch_size
        #     # n_neg=3
        #     roi_feat_all=[]
        #     img_all=[]
        #     scores=[]
        #     # iou_all=best_n_list
        #     roi_batch_all=[]
        #     label_batch_all=[]
        #     lang_all = []
        #
        #     FG_THRESH = 0.9
        #     BG_THRESH_HI = 0.4
        #     BG_THRESH_LO = 0.01
        #     fg_rois_per_image = 2
        #     rois_per_image = 8
        #
        #     bbox=torch.from_numpy(np.array(bbox)).cuda()
        #
        #     for scale_ii in range(len(pred_anchor)):
        #
        #         grid, grid_size = args.size // (32 // (2 ** scale_ii)), 32 // (2 ** scale_ii)
        #         anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
        #         anchors = [anchors_full[i] for i in anchor_idxs]
        #         # scaled_anchors = torch.from_numpy(np.asarray([(x[0] / (args.anchor_imsize / grid), \
        #         #                    x[1] / (args.anchor_imsize / grid)) for x in anchors])).float()
        #
        #         ws = np.asarray([np.round(x[0] * grid_size / (args.anchor_imsize / grid)) for x in anchors])
        #         hs = np.asarray([np.round(x[1] * grid_size / (args.anchor_imsize / grid)) for x in anchors])
        #
        #         x_ctr, y_ctr = (grid_size - 1) * 0.5, (grid_size - 1) * 0.5
        #
        #         scaled_anchors = torch.from_numpy(_mkanchors(ws, hs, x_ctr, y_ctr)).float().cuda()
        #
        #
        #         bbox_deltas = pred_anchor[scale_ii][:,:2,:,:].unsqueeze(1).expand(batch_size,3,2,-1,-1)
        #
        #
        #         feat_height, feat_width = grid, grid
        #         shift_x = np.arange(0, feat_width) * grid_size
        #         shift_y = np.arange(0, feat_height) * grid_size
        #         shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        #         shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
        #                                              shift_x.ravel(), shift_y.ravel())).transpose())
        #         shifts = shifts.contiguous().type_as(bbox_deltas).float()
        #
        #         A = 3
        #         K = shifts.size(0)
        #
        #         # self._anchors = self._anchors.type_as(scores)
        #         # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        #         anchors = scaled_anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        #         anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        #
        #         bbox_deltas = bbox_deltas.permute(0, 1, 3, 4, 2).contiguous()
        #         bbox_deltas = bbox_deltas.view(batch_size, -1, 2)
        #
        #         proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size,grid_size) # xyxy
        #
        #         proposals = clip_boxes(proposals, args.size, batch_size)
        #
        #         gt_boxes = bbox.clone().unsqueeze(1).float() #xyxy
        #         ## make more sample for gt
        #         gt_boxes_more= more_boxes(gt_boxes, fg_rois_per_image, args.size)
        #
        #
        #
        #         # gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        #         # gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        #
        #         # Include ground-truth boxes in the set of candidate rois
        #         all_rois = torch.cat([proposals, gt_boxes_more], 1)
        #
        #         overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        #
        #         max_overlaps, gt_assignment = torch.max(overlaps, 2)
        #
        #         batch_size = overlaps.size(0)
        #         num_proposal = overlaps.size(1)
        #         num_boxes_per_img = overlaps.size(2)
        #
        #         offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        #         offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        #
        #         labels = gt_boxes[:, :, 3]
        #         labels[:, :]=1.
        #         labels = labels.contiguous().view(-1)[offset.view(-1)] \
        #             .view(batch_size, -1)
        #         # labels = torch.ones(batch_size,1).cuda()
        #
        #
        #         # roi_size=(scale_ii+1)*7
        #
        #         labels_batch = labels.new(batch_size, rois_per_image).zero_()
        #         rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        #         lang_batch=[]
        #
        #
        #
        #         for i in range(batch_size):
        #             fg_inds = torch.nonzero(max_overlaps[i] >= FG_THRESH).view(-1)
        #             fg_num_rois = fg_inds.numel()
        #             # print(fg_num_rois)
        #             # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        #             bg_inds = torch.nonzero((max_overlaps[i] < BG_THRESH_HI) &
        #                                     (max_overlaps[i] >= BG_THRESH_LO)).view(-1)
        #             bg_num_rois = bg_inds.numel()
        #
        #             if fg_num_rois > 0 and bg_num_rois > 0:
        #                 # sampling fg
        #                 fg_rois_per_this_image = fg_rois_per_image#min(fg_rois_per_image, fg_num_rois)
        #
        #                 # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
        #                 # See https://github.com/pytorch/pytorch/issues/1868 for more details.
        #                 # use numpy instead.
        #                 # rand_num = torch.randperm(fg_num_rois).long().cuda()
        #                 if fg_rois_per_image <= fg_num_rois:
        #                     rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
        #                     fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
        #                 else:
        #                     rand_num = torch.from_numpy(np.random.choice(fg_num_rois,fg_rois_per_image,replace=True)).type_as(gt_boxes).long()
        #                     fg_inds = fg_inds[rand_num]
        #                 # sampling bg
        #                 bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        #
        #                 # Seems torch.rand has a bug, it will generate very large number and make an error.
        #                 # We use numpy rand instead.
        #                 # rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
        #                 rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
        #                 rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
        #                 bg_inds = bg_inds[rand_num]
        #
        #             elif fg_num_rois > 0 and bg_num_rois == 0:
        #                 # sampling fg
        #                 # rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
        #                 rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
        #                 rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
        #                 fg_inds = fg_inds[rand_num]
        #                 fg_rois_per_this_image = rois_per_image
        #                 bg_rois_per_this_image = 0
        #             elif bg_num_rois > 0 and fg_num_rois == 0:
        #                 # sampling bg
        #                 # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
        #                 rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
        #                 rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
        #
        #                 bg_inds = bg_inds[rand_num]
        #                 bg_rois_per_this_image = rois_per_image
        #                 fg_rois_per_this_image = 0
        #             else:
        #                 raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")
        #
        #                 # The indices that we're selecting (both fg and bg)
        #             keep_inds = torch.cat([fg_inds, bg_inds], 0)
        #
        #             # Select sampled values from various arrays:
        #             labels_batch[i].copy_(labels[i][keep_inds])
        #
        #             # Clamp labels for the background RoIs to 0
        #             if fg_rois_per_this_image < rois_per_image:
        #                 labels_batch[i][fg_rois_per_this_image:] = 0
        #
        #             rois_batch[i,:, 1:] = all_rois[i][keep_inds]
        #             rois_batch[i, :, 0] = i
        #             lang_batch.append(torch.stack([flang[i]] * rois_per_image))
        #         roi_batch_all.append(rois_batch)
        #         label_batch_all.append(labels_batch)
        #         lang_all.append(torch.stack(lang_batch))
        #     # for i in range(batch_size):
        #
        #     roi_batch_all=torch.cat(roi_batch_all)
        #     label_batch_all=torch.cat(label_batch_all)
        #     flang = torch.cat(lang_all)
        #
        #     fvisu = []
        #     for ii in range(len(intmd_fea)):
        #         fvisu.append(self.mapping_visu._modules[str(ii)](intmd_fea[ii]))
        #         fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)
        #
        #
        #     for scale_ii in range(len(intmd_fea)):
        #         grid, grid_size = args.size // (32 // (2 ** scale_ii)), 32 // (2 ** scale_ii)
        #         roi_size = (scale_ii + 1) * 10
        #
        #         feat_map=fvisu[scale_ii]
        #         # roi_scale=torch.cat([roi_batch_all.view(-1, 5)[:,0].unsqueeze(1),roi_batch_all.view(-1, 5)[:,1:]/grid_size],dim=1)
        #
        #         roi_feat=roi_align(feat_map,roi_batch_all.view(-1, 5),[roi_size,roi_size], 1./grid_size)
        #         roi_img=roi_align(image,roi_batch_all.view(-1, 5),[roi_size,roi_size])
        #
        #         roi_feat_all.append(torch.cat([roi_img,roi_feat],dim=1))
        #         # roi_feat_all.append(roi_feat)
        #         img_all.append(roi_img)
        #         scores.append(label_batch_all.view(-1))
        #
        #
        #     cam,cam_rv, bi_score = [], [] , []
        #     for ii in range(len(roi_feat_all)):
        #         # output=self.fcn_out._modules[str(ii)](roi_feat_all[ii])
        #         emb=self.fcn_emb._modules[str(ii)](roi_feat_all[ii])
        #         output=self.fcn_out._modules[str(ii)](emb)
        #         cam.append(output)
        #         cam_rv.append(output)
        #         # cam_rv.append(self.PCM(output, emb, flang, img_all[ii], ii))
        #         bi_score.append(F.adaptive_avg_pool2d(cam[ii], (1,1)).squeeze())
        #
        #     return cam,cam_rv, bi_score, scores,roi_batch_all
        # else:

        (intmd_fea, image, flang, seg_bbox, args)=input
        # print(seg_bbox)
        # fvisu = []
        # for ii in range(len(intmd_fea)):
        #     fvisu.append(self.mapping_visu._modules[str(ii)](intmd_fea[ii]))
        #     fvisu[ii] = F.normalize(fvisu[ii], p=2, dim=1)

        seg_bbox = torch.from_numpy(np.array(seg_bbox)).float().cuda()
        batch_size = seg_bbox.size(0)
        # feats = seg_bbox.unsqueeze(0)
        rois_batch = seg_bbox.new(batch_size, 5).zero_()
        for ii in range(batch_size):
            rois_batch[ii, 1:] = seg_bbox[ii]
            rois_batch[ii, 0] = ii

        roi_feat_all, img_all=[], []
        for scale_ii in range(len(intmd_fea)):
            grid, grid_size = args.size // (32 // (2 ** scale_ii)), 32 // (2 ** scale_ii)
            roi_size = 40 #(scale_ii + 1) * 10
            # for ii in range(batch_size):
             #[x.unsqueeze(0) for x in seg_bbox[scale_ii]]
            feat_map = intmd_fea[scale_ii].detach()
            # if torch.isnan(feat_map.sum()) or torch.isinf(feat_map.sum()):
            #     print("feat_map")
            #     exit(1)
            roi_feat = roi_align(feat_map, rois_batch, [roi_size, roi_size],1./grid_size)
            # if torch.isnan(roi_feat.sum()) or torch.isinf(roi_feat.sum()):
            #     print("roi_feat")
            #     exit(1)
            roi_img=roi_align(image.detach(),rois_batch,[roi_size,roi_size])
            # if torch.isnan(roi_img.sum()) or torch.isinf(roi_img.sum()):
            #     print("roi_img")
            #     exit(1)
            roi_img=F.normalize(roi_img+1e-8)
            roi_feat=F.normalize(roi_feat+1e-8)
            roi_feat_all.append(torch.cat([roi_img,roi_feat],dim=1))
            # roi_feat_all.append(roi_feat)
            img_all.append(roi_img)
        roi_feat_all=torch.stack(roi_feat_all).transpose(1, 0).mean(1)
        # if sum(torch.isinf(roi_feat_all).nonzero())>0:
        #     print("roi_feat_all")
        #     exit(1)
        cam,cam_rv, bi_score = [], [] , []
        roi_feat_all = F.normalize(roi_feat_all + 1e-8)
        # for ii in range(len(roi_feat_all)):
            # output=self.fcn_out._modules[str(ii)](roi_feat_all[ii])
        emb=self.fcn_emb(roi_feat_all)
        cam=self.fcn_out(emb)
        # cam.append(output)
        # cam_rv.append(output)
        # cam_rv.append(self.PCM(output,emb,flang, img_all[ii], ii))
        bi_score=F.adaptive_avg_pool2d(cam, (1,1)).squeeze()

        return cam, cam_rv, bi_score

    def PCM(self, cam, f, lang,x, scale_ii):
        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0


        # n, c, h, w = f.size()
        # cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)


        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f.detach()], dim=1)
        f = self.refine._modules[str(scale_ii)](f)
        # f = self.bilinear_att(f,lang)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam_d_norm.view(n,-1,h*w), aff).view(n, -1, h, w)

        return cam_rv

    def bilinear_att(self, f, lang):

        n, c, h, w = f.size()
        f = f.view(n,-1,h*w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5) # n x c x hw

        lang=self.compress_lang(lang.view(n,-1)).view(n,1,-1) # n x 1 x c

        aff = F.relu(torch.matmul(lang, f), inplace=True) # n x 1 x hw
        aff = aff / (torch.sum(aff, dim=2, keepdim=True) + 1e-5)
        cam_rv = f*aff # +f #torch.matmul(f, aff)


        return cam_rv





