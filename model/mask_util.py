import torch
import torch.nn as nn
from utils.utils import *
from pydensecrf import densecrf
from scipy import ndimage
import numpy as np


def crf_inference(sigm_val, H, W, proc_im):

    sigm_val = np.squeeze(sigm_val)
    d = densecrf.DenseCRF2D(W, H, 2)
    U = np.expand_dims(-np.log(sigm_val + 1e-8), axis=0)
    U_ = np.expand_dims(-np.log(1 - sigm_val + 1e-8), axis=0)
    unary = np.concatenate((U_, U), axis=0)
    unary = unary.reshape((2, -1))
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
    Q = d.inference(5)
    pred_raw_dcrf = np.argmax(Q, axis=0).reshape((H, W)).astype(np.float32)
    # predicts_dcrf = im_processing.resize_and_crop(pred_raw_dcrf, mask.shape[0], mask.shape[1])

    return pred_raw_dcrf


def max_norm(p, version='torch', e=1e-5):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
            min_v = torch.min(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
            p = F.relu(p - min_v - e) / (max_v - min_v + e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
            min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
            p = F.relu(p - min_v - e) / (max_v - min_v + e)
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p < 0] = 0
            max_v = np.max(p, (1, 2), keepdims=True)
            min_v = np.min(p, (1, 2), keepdims=True)
            p[p < min_v + e] = 0
            p = (p - min_v - e) / (max_v + e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p < 0] = 0
            max_v = np.max(p, (2, 3), keepdims=True)
            min_v = np.min(p, (2, 3), keepdims=True)
            p[p < min_v + e] = 0
            p = (p - min_v - e) / (max_v + e)
    return p

def load_cam(cams,b, n, k, gj, gi,  args, crf=False, crf_img=None):

    pred_ch = gj * k + gi
    cam = cams[n][b, pred_ch]
    cam = max_norm(cam.unsqueeze(0).unsqueeze(0))
    cam = F.interpolate(cam, (args.size, args.size), mode='bilinear', align_corners=True)
    cam = F.relu(cam)

    # cam = run_pamr(imgs, cam)

    if crf:
        cam = cam.squeeze().data.cpu().numpy()
        # cam = cam - np.min(cam)
        # cam = cam / (np.max(cam)+1e-8)
        cam = crf_inference(cam, args.size, args.size, crf_img)
        # cam=cv2.resize(cam, (256,256), interpolation=cv2.INTER_CUBIC)
        cam = np.stack([cam] * 3).transpose(1, 2, 0)
    else:
        cam = cam.squeeze().data.cpu().numpy()

        cam = cam - np.min(cam)
        cam = cam / (np.max(cam ) +1e-8)


    return cam


def extract_bboxes(mask, center_gt,bbox_gt):
    label_im, nb_labels = ndimage.label(mask)

    # Find the largest connected component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    # return bbox_gt
    # Now that we have only one connected component, extract it's bounding box
    slice_xy = ndimage.find_objects(label_im)
    points = []
    for slice in slice_xy:
        points.append([int(slice[1].start), int(slice[0].start), int(slice[1].stop), int(slice[0].stop)])

    for pp in points:
        if (pp[0] <= center_gt[0] <= pp[2]) and (pp[1] <= center_gt[1] <= pp[3]):
            return pp

    if len(points)<1:
        return bbox_gt
    return points[0]


def box_predict(cams,gj,gi,center_gt,bbox_gt,args):

    batch_size=args.batch_size
    cam_masks=[]
    coord_batch=[]
    for ii in range(batch_size):
        cam_batch=[]
        for scale_ii in range(len(cams)):
            cam_batch.append(load_cam(cams,ii,scale_ii,int( args.size // (32 // (2 ** scale_ii))),gj[scale_ii][ii],gi[scale_ii][ii],args))
        cam4 = (cam_batch[0] + cam_batch[1] + cam_batch[2]) / 3
        cam4 = cam4 - np.min(cam4)
        cam4 = cam4 / np.max(cam4 + 1e-8)
        cam4[cam4 < 0.2] = 0
        # cam_box[cam_box > 0] = 1
        coord_batch.append(extract_bboxes(cam4, center_gt[ii],bbox_gt[ii]))
        cam_masks.append(torch.from_numpy(cam4))

    return coord_batch,cam_masks