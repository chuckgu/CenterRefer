import os
import sys
import argparse
import shutil
import time
import random
import gc
import json
from distutils.version import LooseVersion
import scipy.misc
import logging


import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from utils.transforms import ResizeImage, ResizeAnnotation

from dataset.referit_loader import *
from model.grounding_model import *
from utils.parsing_metrics import *
from utils.utils import *
from tqdm import tqdm


def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U

def compute_dists(preds, targets, thr):
    dists = np.zeros((preds.shape[0]))
    for n in range(preds.shape[0]):
        normed_preds = preds[n]
        normed_targets = targets[n]
        dists[n] = np.linalg.norm(normed_preds - normed_targets)

    return np.less(dists, thr).sum() * 1.0

def compute_point_box(preds,bbox):
    inout = np.zeros((preds.shape[0]))
    for n in range(preds.shape[0]):
        normed_preds = preds[n]
        normed_bbox = bbox[n]
        inout[n] = (normed_bbox[0]<=normed_preds[0]<=normed_bbox[2]) and (normed_bbox[1]<=normed_preds[1]<=normed_bbox[3])

    return inout.sum() * 1.0

def vis_detections(im, class_name, dets, color, thresh=0.0):
    """Visual debugging of detections."""
    # for i in range(np.minimum(10, dets.shape[0])):

    bbox = tuple(int(np.round(x)) for x in dets[:4])
    score = 1
    cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
    cv2.putText(im, '%s' % (class_name), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 255), thickness=1)

    return im

def max_norm(p, version='torch', e=1e-5):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e)/(max_v-min_v+e)
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<0] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p[p<min_v+e] = 0
			p = (p-min_v-e)/(max_v+e)
	return p

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def yolo_loss(input, target, gi, gj,  w_coord=5., w_neg=1./5, size_average=True):
    mseloss = torch.nn.MSELoss(size_average=True)
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    batch = input[0].size(0)


    loss_x,loss_y=0.,0.
    for scale_ii in range(len(input)):
        pred_bbox = Variable(torch.zeros(batch, 2).cuda())
        gt_bbox = Variable(torch.zeros(batch, 2).cuda())
        for ii in range(batch):
            pred_bbox[ii, :] = F.sigmoid(input[scale_ii][ii,:2,gj[scale_ii][ii],gi[scale_ii][ii]])
            # pred_bbox[ii, 2:4] = input[best_n_list[ii]//3][ii,best_n_list[ii]%3,2:4,gj[ii],gi[ii]]
            gt_bbox[ii, :] = target[scale_ii][ii,:2,gj[scale_ii][ii],gi[scale_ii][ii]]
        loss_x += mseloss(pred_bbox[:,0], gt_bbox[:,0])
        loss_y += mseloss(pred_bbox[:,1], gt_bbox[:,1])
    # loss_w = mseloss(pred_bbox[:,2], gt_bbox[:,2])
    # loss_h = mseloss(pred_bbox[:,3], gt_bbox[:,3])

    # pred_conf_list, gt_conf_list = [], []
    loss_conf=0.
    for scale_ii in range(len(input)):
        pred_conf=input[scale_ii][:,2,:,:].contiguous().view(batch,-1)
        gt_conf=target[scale_ii][:,2,:,:].contiguous().view(batch,-1)
        loss_conf+=celoss(pred_conf, gt_conf.max(1)[1])
        # pred_conf_list.append(input[scale_ii][:,2,:,:].contiguous().view(batch,-1))
        # gt_conf_list.append(target[scale_ii][:,2,:,:].contiguous().view(batch,-1))
    # pred_conf = torch.cat(pred_conf_list, dim=0)
    # gt_conf = torch.cat(gt_conf_list, dim=0)
    # loss_conf = celoss(pred_conf, gt_conf.max(1)[1])
    return (loss_x+loss_y)*w_coord/3 + loss_conf/3

def refine_loss(input, target):
    celoss = torch.nn.CrossEntropyLoss(size_average=True)
    input = torch.cat(input)
    target = torch.cat(target).long()

    return celoss(input, target)

def seam_loss(cam1, cam_rv1,cam2, cam_rv2, label_ ):
    N=cam1.size(0)
    bg_score = torch.ones((N, 1)).cuda()
    label = torch.cat((bg_score, label_.unsqueeze(1)), dim=1)
    label = label.unsqueeze(2).unsqueeze(3)

    label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
    # loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])
    cam1 = max_norm(cam1) * label
    cam_rv1 = max_norm(cam_rv1) * label

    label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
    # loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
    cam2 = max_norm(cam2) * label
    cam_rv2 = max_norm(cam_rv2) * label

    # loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
    # loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])


    loss_cls1 = F.cross_entropy(label1.squeeze(3).squeeze(2), label_.long())
    loss_cls2 = F.cross_entropy(label2.squeeze(3).squeeze(2), label_.long())


    ns, cs, hs, ws = cam2.size()
    loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))
    # loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
    cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
    cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]
    #            with torch.no_grad():
    #                eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
    tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
    tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
    loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(2 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(2 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr = loss_ecr1 + loss_ecr2

    loss_cls = (loss_cls1 + loss_cls2) / 2 #+ (loss_rvmin1 + loss_rvmin2) / 2
    loss = loss_cls + loss_er + loss_ecr

    return loss


def save_segmentation_map(bbox, target_bbox, input, mode, batch_start_index, \
    merge_pred=None, pred_conf_visu=None, save_path='./visulizations/'):
    n = input.shape[0]
    save_path=save_path+mode

    input=input.data.cpu().numpy()
    input=input.transpose(0,2,3,1)
    for ii in range(n):
        os.system('mkdir -p %s/sample_%d'%(save_path,batch_start_index+ii))
        imgs = input[ii,:,:,:].copy()
        imgs = (imgs*np.array([0.299, 0.224, 0.225])+np.array([0.485, 0.456, 0.406]))*255.
        # imgs = imgs.transpose(2,0,1)
        imgs = np.array(imgs, dtype=np.float32)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.rectangle(imgs, (bbox[ii,0], bbox[ii,1]), (bbox[ii,2], bbox[ii,3]), (255,0,0), 2)
        cv2.rectangle(imgs, (target_bbox[ii,0], target_bbox[ii,1]), (target_bbox[ii,2], target_bbox[ii,3]), (0,255,0), 2)
        cv2.imwrite('%s/sample_%d/pred_yolo.png'%(save_path,batch_start_index+ii),imgs)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
    if args.power!=0.:
        lr = lr_poly(args.lr, i_iter, args.nb_epoch, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
      optimizer.param_groups[1]['lr'] = lr / 10
        
def save_checkpoint(state, is_best, filename='default'):
    if filename=='default':
        filename = 'model_%s_batch%d'%(args.dataset,args.batch_size)

    checkpoint_name = './saved_models/%s_checkpoint.pth.tar'%(filename)
    best_name = './saved_models/%s_model_best.pth.tar'%(filename)
    torch.save(state, checkpoint_name)
    if is_best:
        shutil.copyfile(checkpoint_name, best_name)

def build_target(raw_coord, pred):

    coord_list, bbox_list = [],[]
    best_gi, best_gj = [], []
    for scale_ii in range(len(pred)): # change gt box coord to coor in feat map
        coord = Variable(torch.zeros(raw_coord.size(0), raw_coord.size(1)).cuda())
        batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii)) # 1/32, 1/16, 1/8
        coord[:, 0] = raw_coord[:,0] / args.size
        coord[:, 1] = raw_coord[:, 1] / args.size

        # coord[:,0] = (raw_coord[:,0] + raw_coord[:,2])/(2*args.size)
        # coord[:,1] = (raw_coord[:,1] + raw_coord[:,3])/(2*args.size)
        # coord[:,2] = (raw_coord[:,2] - raw_coord[:,0])/(args.size)
        # coord[:,3] = (raw_coord[:,3] - raw_coord[:,1])/(args.size)
        coord = coord * grid
        coord_list.append(coord)
        bbox_list.append(torch.zeros(coord.size(0),3,grid, grid))
        best_gi.append([])
        best_gj.append([])

    # best_n_list, best_gi, best_gj = [],[],[]
    # neg_gi, neg_gj=[],[]


    # iou_all=np.zeros((len(pred),batch,3))
    for ii in range(batch):
        # best_gi = []
        # best_gj=[]
        for scale_ii in range(len(pred)):
            batch, grid = raw_coord.size(0), args.size//(32//(2**scale_ii))
            gi = coord_list[scale_ii][ii,0].long()
            gj = coord_list[scale_ii][ii,1].long()
            tx = coord_list[scale_ii][ii,0] - gi.float()
            ty = coord_list[scale_ii][ii,1] - gj.float()

            # gw = coord_list[scale_ii][ii,2]
            # gh = coord_list[scale_ii][ii,3]
            #
            # anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
            # anchors = [anchors_full[i] for i in anchor_idxs]
            # scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
            #     x[1] / (args.anchor_imsize/grid)) for x in anchors]
            #
            # ## Get shape of gt box
            # gt_box = torch.from_numpy(np.array([0, 0, gw, gh]).astype(np.float32)).unsqueeze(0)
            # ## Get shape of anchor box
            # anchor_shapes = torch.from_numpy(np.concatenate((np.zeros((len(scaled_anchors), 2)), np.array(scaled_anchors)), 1).astype(np.float32))
            ## Calculate iou between gt and anchor shapes

            bbox_list[scale_ii][ii, :, gj, gi] = torch.stack(
                [tx, ty, torch.ones(1).cuda().squeeze()])
            best_gi[scale_ii].append(gi)
            best_gj[scale_ii].append(gj)

            # iou_list=list(bbox_iou(gt_box, anchor_shapes))
            # anch_ious += iou_list
            # iou_all[scale_ii,ii,:]=np.array(iou_list)
        # ## Find the best matching anchor box
        # best_n = np.argmax(np.array(anch_ious)) # select best match anchor box
        # best_scale = best_n//3 # select best match scaled feature map
        #
        # batch, grid = raw_coord.size(0), args.size//(32/(2**best_scale))
        # anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
        # anchors = [anchors_full[i] for i in anchor_idxs]
        # scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
        #     x[1] / (args.anchor_imsize/grid)) for x in anchors]
        #
        # gi = coord_list[best_scale][ii,0].long()
        # gj = coord_list[best_scale][ii,1].long()
        # tx = coord_list[best_scale][ii,0] - gi.float()
        # ty = coord_list[best_scale][ii,1] - gj.float()
        # gw = coord_list[best_scale][ii,2]
        # gh = coord_list[best_scale][ii,3]
        # tw = torch.log(gw / scaled_anchors[best_n%3][0] + 1e-16)
        # th = torch.log(gh / scaled_anchors[best_n%3][1] + 1e-16)
        #
        # bbox_list[best_scale][ii, best_n%3, :, gj, gi] = torch.stack([tx, ty, tw, th, torch.ones(1).cuda().squeeze()])
        # best_n_list.append(int(best_n))
        # best_gi.append(gi)
        # best_gj.append(gj)
        # # iou_all.append(iou_scale)


    for ii in range(len(bbox_list)):
        bbox_list[ii] = Variable(bbox_list[ii].cuda())
    return bbox_list, best_gi, best_gj

def main():
    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--workers', default=16, type=int, help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--batch_size', default=28, type=int, help='batch size')
    parser.add_argument('--size_average', dest='size_average', 
                        default=False, action='store_true', help='size_average')
    parser.add_argument('--size', default=256, type=int, help='image size')
    parser.add_argument('--anchor_imsize', default=416, type=int,
                        help='scale used to calculate anchors defined in model cfg file')
    parser.add_argument('--data_root', type=str, default='./ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='data',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--dataset', default='unc', type=str,
                        help='referit/flickr/unc/unc+/gref')
    parser.add_argument('--time', default=20, type=int,
                        help='maximum time steps (lang length) per batch')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='/shared/CenterCam/saved_models/Center_cam_aff_checkpoint.pth.tar', type=str, metavar='PATH',# bert_unc_model.pth.tar,/shared/ReferCam/saved_models/ReferCam_model_best.pth.tar
                        help='pretrain support load state_dict that are not identical, while have no loss saved as resume')
    parser.add_argument('--optimizer', default='RMSprop', help='optimizer: sgd, adam, RMSprop')
    parser.add_argument('--print_freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='ReferPoint_unc', type=str, help='Name head for saved model')
    parser.add_argument('--save_plot', dest='save_plot', default=False, action='store_true', help='save visulization plots')
    parser.add_argument('--seed', default=13, type=int, help='random seed')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--test', dest='test', default=False, action='store_true', help='test')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--lstm', dest='lstm', default=False, action='store_true', help='if use lstm as language module instead of bert')
    parser.add_argument('--seg', dest='seg', default=False, action='store_true', help='if use lstm as language module instead of bert')

    global args, anchors_full
    args = parser.parse_args()
    print('----------------------------------------------------------------------')
    print(sys.argv[0])
    print(args)
    print('----------------------------------------------------------------------')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    
    ## fix seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed+1)
    torch.manual_seed(args.seed+2)
    torch.cuda.manual_seed_all(args.seed+3)

    eps=1e-10
    anchors_full=get_archors_full(args)

    ## save logs
    if args.savename=='default':
        args.savename = 'model_%s_batch%d'%(args.dataset,args.batch_size)
    if not os.path.exists('./logs'):
        os.mkdir('logs')
    logging.basicConfig(level=logging.DEBUG, filename="./logs/%s"%args.savename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='train',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm,
                         augment=True)
    val_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split='val',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm)
    ## note certain dataset does not have 'test' set:
    ## 'unc': {'train', 'val', 'trainval', 'testA', 'testB'}
    test_dataset = ReferDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         testmode=True,
                         split='testA',
                         imsize = args.size,
                         transform=input_transform,
                         max_query_len=args.time,
                         lstm=args.lstm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              pin_memory=True, drop_last=True, num_workers=0)

    ## Model
    ## input ifcorpus=None to use bert as text encoder
    ifcorpus = None
    if args.lstm:
        ifcorpus = train_dataset.corpus
    model = grounding_model(corpus=ifcorpus, light=args.light, emb_size=args.emb_size, coordmap=True,\
        bert_model=args.bert_model, dataset=args.dataset, seg=args.seg)
    # model=model.cuda()

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'segmentation' not in k}
            assert (len([k for k, v in pretrained_dict.items()])!=0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=True)
            print("=> loaded pretrain model at {}"
                  .format(args.pretrain))
            logging.info("=> loaded pretrain model at {}"
                  .format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss)))
            logging.info("=> loaded checkpoint (epoch {}) Loss{}"
                  .format(checkpoint['epoch'], best_loss))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))




    print('Num of parameters:', sum([param.nelement() for param in model.parameters()]))
    logging.info('Num of parameters:%d'%int(sum([param.nelement() for param in model.parameters()])))

    visu_param = model.module.visumodel.parameters()
    rest_param = [param for param in model.parameters() if param not in visu_param]
    visu_param = list(model.module.visumodel.parameters())
    sum_visu = sum([param.nelement() for param in visu_param])
    sum_text = sum([param.nelement() for param in model.module.textmodel.parameters()])
    sum_fusion = sum([param.nelement() for param in rest_param]) - sum_text
    print('visu, text, fusion module parameters:', sum_visu, sum_text, sum_fusion)

    # if args.seg:
    #     param=model.module.segmentation.parameters()
    # else:
    #     param=model.parameters()



    ## optimizer; rmsprop default
    if args.optimizer=='adam':
        optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=0.0005)
    elif args.optimizer=='sgd':
        optimizer = torch.optim.SGD(param, lr=args.lr, momentum=0.99)
    else:
        optimizer = torch.optim.RMSprop([{'params': rest_param},
                {'params': visu_param, 'lr': args.lr/10.}], lr=args.lr, weight_decay=0.0005)

    # print([name for name, param in model.named_parameters() if param not in model.module.visumodel.parameters()])


    ## training and testing
    best_accu = -float('Inf')
    accu_new = validate_epoch(val_loader, model, args.size_average)
    if args.test:
        _ = test_epoch(test_loader, model, args.size_average)
        exit(0)
    for epoch in range(args.nb_epoch):
        adjust_learning_rate(optimizer, epoch)
        train_epoch(train_loader, model, optimizer, epoch, args.size_average)
        accu_new = validate_epoch(val_loader, model, args.size_average)
        ## remember best accu and save checkpoint
        is_best = accu_new > best_accu
        best_accu = max(accu_new, best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': accu_new,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=args.savename)
    print('\nBest Accu: %f\n'%best_accu)
    logging.info('\nBest Accu: %f\n'%best_accu)

def get_args():
    return args

def train_epoch(train_loader, model, optimizer, epoch, size_average):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_losses=AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()
    acc_refine = AverageMeter()

    model.train()
    end = time.time()
    tbar=tqdm(train_loader)

    # for batch_idx, (imgs, word_id, word_mask, bbox) in enumerate(train_loader):
    for batch_idx, (imgs, word_id, word_mask, bbox, mask, center) in enumerate(tbar):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(center)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        ## Note LSTM does not use word_mask
        pred_anchor,intmd_fea, flang = model(image, word_id, word_mask)

        ## convert gt box to center+offset format
        gt_param, gi, gj = build_target(bbox, pred_anchor)
        ## flatten anchor dim at each scale
        # pred_conf_list=[]
        # for ii in range(len (pred_anchor)):
        #     pred_anchor[ii] = pred_anchor[ii].view(pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        #     # pred_conf_list.append(pred_anchor[ii][:, :, 4, :, :].contiguous().view(args.batch_size, -1))
        pred_conf_list = []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,2,:,:].contiguous().view(args.batch_size,-1))
        if args.seg:
            cam, cam_rv, bi_score, gt_score = model.module.segmentation((intmd_fea,image,flang, bbox, pred_anchor, args))
            # n*3*8 x 2 x 20 x 20


        ## training offset eval: if correct with gt center loc
        ## convert offset pred to boxes
        pred_coord = torch.zeros(len(pred_anchor),args.batch_size,2)
        for scale_ii in range(len(pred_anchor)):
            for ii in range(args.batch_size):

                grid, grid_size = args.size//(32//(2**scale_ii)), 32//(2**scale_ii)
                # anchor_idxs = [x + 3*best_scale_ii for x in [0,1,2]]
                # anchors = [anchors_full[i] for i in anchor_idxs]
                # scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                #     x[1] / (args.anchor_imsize/grid)) for x in anchors]

                pred_coord[scale_ii,ii,0] = F.sigmoid(pred_anchor[scale_ii][ii, 0, gj[scale_ii][ii], gi[scale_ii][ii]]) + gi[scale_ii][ii].float()
                pred_coord[scale_ii,ii,1] = F.sigmoid(pred_anchor[scale_ii][ii, 1, gj[scale_ii][ii], gi[scale_ii][ii]]) + gj[scale_ii][ii].float()
                pred_coord[scale_ii,ii,:] = pred_coord[scale_ii,ii,:] * grid_size
        # pred_coord = xywh2xyxy(pred_coord)

        ## loss
        ref_loss=0.
        if args.seg:
            ref_loss = (seam_loss(cam[0],cam_rv[0],cam[1],cam_rv[1],gt_score[0])+seam_loss(cam[1],cam_rv[1],cam[2],cam_rv[2],gt_score[1])+seam_loss(cam[2],cam_rv[2],cam[0],cam_rv[0],gt_score[2]))/3
            # ref_loss=refine_loss(bi_score, gt_score)
        loss = yolo_loss(pred_anchor, gt_param, gi, gj) + ref_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))

        if args.seg:
            seg_losses.update(ref_loss.item(), imgs.size(0))
            rois_per_image = 8
            for ii in range(len(bi_score)):
                accr=np.sum(np.array(bi_score[ii].max(1)[1].data.cpu().numpy()== gt_score[ii].data.cpu().numpy(),dtype=float))/args.batch_size/rois_per_image/3
                acc_refine.update(accr, imgs.size(0)*rois_per_image*3)

        ## box iou
        target_bbox = center
        # for ii in range(len(pred_coord)):
        #     torch.nn.functional.pairwise_distance(pred_coord[ii], target_bbox, p=2.0)
        # iou = bbox_iou(pred_coord, target_bbox.data.cpu(), x1y1x2y2=True)
        # accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        ## evaluate if center location is correct
        # pred_conf_list, gt_conf_list = [], []
        accu_center=0.
        for ii in range(len(pred_anchor)):
            # pred_conf_list.append(pred_anchor[ii][:,2,:,:].contiguous().view(args.batch_size,-1))
            # gt_conf_list.append(gt_param[ii][:,2,:,:].contiguous().view(args.batch_size,-1))
            pred_conf = pred_anchor[ii][:,2,:,:].contiguous().view(args.batch_size,-1)
            gt_conf = gt_param[ii][:,2,:,:].contiguous().view(args.batch_size,-1)
            accu_center += np.sum((pred_conf.max(1)[1] == gt_conf.max(1)[1]).cpu().numpy().astype(np.float32))/args.batch_size

        ## metrics
        # accu_center=0.
        miou.update(0, imgs.size(0))
        acc.update(0, imgs.size(0))
        acc_center.update(accu_center/3, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tbar.set_description(
            'Loss {seg_loss.avg:.4f} '\
                'Accu_grid {acc_c.avg:.4f} ' \
            .format(seg_loss=losses, acc_c=acc_center))

        if args.save_plot:
            # if batch_idx%100==0 and epoch==args.nb_epoch-1:
            if True:
                save_segmentation_map(pred_coord,target_bbox,imgs,'train',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)

        if (batch_idx+1) % args.print_freq == 0 or (batch_idx+1)==len(train_loader):
            print_str = '\rEpoch: [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                .format( \
                    epoch, (batch_idx+1), len(train_loader), batch_time=batch_time, \
                    data_time=data_time, loss=losses, seg_loss=seg_losses, miou=miou, acc=acc_center, acc_c=acc_refine)
            print(print_str, end="\n")
            # print('\n')
            logging.info(print_str)


def validate_epoch(val_loader, model, size_average, mode='val'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_refine= AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()

    model.eval()
    end = time.time()
    tbar = tqdm(val_loader)

    for batch_idx, (imgs, word_id, word_mask, bbox_gt, mask, center) in enumerate(tbar):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        center = center.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(center)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            ## Note LSTM does not use word_mask
            pred_anchor,intmd_fea,flang = model(image, word_id, word_mask)
        # for ii in range(len(pred_anchor)):
        #     pred_anchor[ii] = pred_anchor[ii].view(   \
        #             pred_anchor[ii].size(0),3,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj= build_target(bbox, pred_anchor)


        ## eval: convert center+offset to box prediction
        ## calculate at rescaled image during validation for speed-up
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,2,:,:].contiguous().view(args.batch_size,-1))
            gt_conf_list.append(F.softmax(gt_param[ii][:,2,:,:].contiguous().view(args.batch_size,-1),dim=1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(args.batch_size,4)
        seg_bbox= torch.zeros(len(intmd_fea),args.batch_size,4).cuda()
        pred_gi, pred_gj, pred_best_n = [],[],[]
        target_best_gi,target_best_gj=[],[]
        for ii in range(args.batch_size):
            if max_loc[ii] < 1*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 1*(args.size//32)**2 + 1*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(args.batch_size,1,grid,grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            target_best_gi.append(target_gi[best_scale][ii])
            target_best_gj.append(target_gj[best_scale][ii])
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, 1, gj, gi]) + gj
            # pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            # pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]

            # for scale_ii in range(len(intmd_fea)):
            #     grid_ratio = (2 ** (scale_ii - best_scale))
            #     seg_bbox[scale_ii,ii,:]=pred_bbox[ii,:] * grid_ratio
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        # pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = center
        if args.seg:
            cam, cam_rv, bi_score = model.module.segmentation((intmd_fea,image, flang, pred_bbox.cuda(), args))


        ## metrics
        iou = 0 #bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)

        accu_center = compute_point_box(np.array(pred_bbox),bbox_gt.data.cpu().numpy())#compute_dists(np.array(pred_bbox),bbox.data.cpu().numpy(),5)/args.batch_size#np.sum(np.array((np.array(target_best_gi) == np.array(pred_gi)) * (np.array(target_best_gj) == np.array(pred_gj)), dtype=float))/args.batch_size

        # accu_center = np.sum(np.array((np.array(target_best_gi) == np.array(pred_gi)) * (np.array(target_best_gj) == np.array(pred_gj)), dtype=float))/args.batch_size
        accu = 0#np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        # gt_onehot=np.array((iou.data.cpu().numpy()>0.5),dtype=float)

        # if args.seg:
        #     for ii in range(len(bi_score)):
        #         accr=np.sum(np.array(bi_score[ii].max(1)[1].data.cpu().numpy()== gt_onehot,dtype=float))/args.batch_size
        #         acc_refine.update(accr, imgs.size(0))

        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(0, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tbar.set_description(
            'Acc in box {acc_refine.val:.4f} ({acc_refine.avg:.4f})'
            .format(acc_refine=acc_center))
        if args.save_plot:
            if batch_idx%1==0:
                save_segmentation_map(pred_bbox,target_bbox,imgs,'val',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)
        
        if (batch_idx+1) % args.print_freq == 0 or (batch_idx+1)==len(val_loader):
            print_str = '\r[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Accu in box {acc.val:.4f} ({acc.avg:.4f})\t' \
                .format( \
                    batch_idx+1, len(val_loader), batch_time=batch_time, \
                    acc=acc_center, acc_c=acc_refine, miou=miou)
            print(print_str, end="\n")
            logging.info(print_str)


    return acc_center.avg

def test_epoch(val_loader, model, size_average, mode='test'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    acc_center = AverageMeter()
    miou = AverageMeter()
    acc_refine = AverageMeter()

    model.eval()
    end = time.time()

    IU_result = list()
    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    mean_IoU, mean_dcrf_IoU = 0, 0
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0.

    tbar = tqdm(val_loader)

    # for batch_idx, (imgs, word_id, word_mask, bbox, mask) in enumerate(tbar):

    for batch_idx, (imgs, word_id, word_mask, bbox_gt, ratio, dw, dh, im_id, mask, center) in enumerate(tbar):
        imgs = imgs.cuda()
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        center = center.cuda()
        image = Variable(imgs)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(center)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            ## Note LSTM does not use word_mask
            pred_anchor,intmd_fea,flang = model(image, word_id, word_mask)
        # for ii in range(len(pred_anchor)):
        #     pred_anchor[ii] = pred_anchor[ii].view(   \
        #             pred_anchor[ii].size(0),3,5,pred_anchor[ii].size(2),pred_anchor[ii].size(3))
        gt_param, target_gi, target_gj = build_target(bbox, pred_anchor)



        ## test: convert center+offset to box prediction
        pred_conf_list, gt_conf_list = [], []
        for ii in range(len(pred_anchor)):
            pred_conf_list.append(pred_anchor[ii][:,2,:,:].contiguous().view(args.batch_size,-1))
            gt_conf_list.append(F.softmax(gt_param[ii][:,2,:,:].contiguous().view(args.batch_size,-1),dim=1))

        pred_conf = torch.cat(pred_conf_list, dim=1)
        gt_conf = torch.cat(gt_conf_list, dim=1)
        max_conf, max_loc = torch.max(pred_conf, dim=1)

        pred_bbox = torch.zeros(1,2)
        target_best_gi,target_best_gj=[],[]
        pred_gi, pred_gj, pred_best_n = [],[],[]
        for ii in range(1):
            if max_loc[ii] < 1*(args.size//32)**2:
                best_scale = 0
            elif max_loc[ii] < 1*(args.size//32)**2 + 1*(args.size//16)**2:
                best_scale = 1
            else:
                best_scale = 2

            grid, grid_size = args.size//(32//(2**best_scale)), 32//(2**best_scale)
            anchor_idxs = [x + 3*best_scale for x in [0,1,2]]
            anchors = [anchors_full[i] for i in anchor_idxs]
            scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                x[1] / (args.anchor_imsize/grid)) for x in anchors]

            pred_conf = pred_conf_list[best_scale].view(args.batch_size,1,grid,grid).data.cpu().numpy()
            max_conf_ii = max_conf.data.cpu().numpy()

            # print(max_conf[ii],max_loc[ii],pred_conf_list[best_scale][ii,max_loc[ii]-64])
            (best_n, gj, gi) = np.where(pred_conf[ii,:,:,:] == max_conf_ii[ii])
            best_n, gi, gj = int(best_n[0]), int(gi[0]), int(gj[0])
            pred_gi.append(gi)
            pred_gj.append(gj)
            target_best_gi.append(target_gi[best_scale][ii])
            target_best_gj.append(target_gj[best_scale][ii])
            pred_best_n.append(best_n+best_scale*3)

            pred_bbox[ii,0] = F.sigmoid(pred_anchor[best_scale][ii, 0, gj, gi]) + gi
            pred_bbox[ii,1] = F.sigmoid(pred_anchor[best_scale][ii, 1, gj, gi]) + gj
            # pred_bbox[ii,2] = torch.exp(pred_anchor[best_scale][ii, best_n, 2, gj, gi]) * scaled_anchors[best_n][0]
            # pred_bbox[ii,3] = torch.exp(pred_anchor[best_scale][ii, best_n, 3, gj, gi]) * scaled_anchors[best_n][1]
            pred_bbox[ii,:] = pred_bbox[ii,:] * grid_size
        # pred_bbox = xywh2xyxy(pred_bbox)
        target_bbox = center

        point_vis=True

        if point_vis:
            dst = imgs.data.cpu().numpy().transpose(0,2,3,1)[0]
            mst = mask.squeeze().data.cpu().numpy()
            center_gt=center.squeeze().data.cpu().numpy()
            gt_box = bbox_gt.squeeze().data.cpu().numpy()
            dst = dst.copy()

            dst = (dst * np.array([0.299, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.
            # dst = dst.transpose(2,0,1)
            dst = np.array(dst, dtype=np.float32)
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            mst = np.stack([mst * 255] * 3).transpose(1, 2, 0)
            point=pred_bbox[0,:2].long().data.cpu().numpy().tolist()
            dst= cv2.addWeighted(dst, 0.7, mst, 0.3,0)

            dst = vis_detections(dst, "GT", gt_box, (204, 0, 0))
            cv2.circle(dst, (center_gt[0], center_gt[1]), 5, (0, 0, 204), -1)
            cv2.circle(dst,(point[0],point[1]),3,(0, 204, 0),-1)

            # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            cv2.imwrite('/shared/CenterCam/cam_out/' + str(im_id[0].split(".")[0]) + ".jpg", dst)
            # Image.fromarray(dst.astype(np.uint8)).save(
            #     '/shared/CenterCam/cam_out/' + str(im_id[0].split(".")[0]) + ".jpg")

        if args.seg:
            cam,_, bi_score = model.module.segmentation((intmd_fea,image, flang, pred_bbox.cuda(), args))

            vis=False
            if vis:

                dst=imgs.squeeze().data.cpu().numpy().transpose(1,2,0)
                mst=mask.squeeze().data.cpu().numpy()
                gt_box=bbox.squeeze().data.cpu().numpy()

                # imgs = input[ii, :, :, :].copy()
                dst = (dst * np.array([0.299, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.
                # imgs = imgs.transpose(2,0,1)
                dst = np.array(dst, dtype=np.float32)
                dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

                dst = vis_detections(dst, "GT", gt_box, (204,0,0))
                dst = vis_detections(dst, "Pred", pred_bbox.squeeze().numpy(), (0, 204, 0))
                def vis_cam(cams,n,h,w,color):
                    height=256
                    # color = (123.7, 116.3, 103.5)
                    ratio = float(height) / max([h,w])
                    new_shape = (round(w * ratio), round(h * ratio))

                    # h,w =256 ,256
                    cam =np.stack([cams[n][:,1].squeeze().data.cpu().numpy()]*3).transpose(1,2,0)
                    cam = cam - np.min(cam)
                    cam_img = cam / np.max(cam)
                    cam_img = np.uint8(255 * cam_img)

                    dw = (height - new_shape[0]) / 2  # width padding
                    dh = (height - new_shape[1]) / 2  # height padding
                    top, bottom = round(dh - 0.1), round(dh + 0.1)
                    left, right = round(dw - 0.1), round(dw + 0.1)
                    cam_img = cv2.resize(cam_img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
                    cam_img = cv2.copyMakeBorder(cam_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

                    # cam_img = cv2.resize(cam_img, (h,w), interpolation=cv2.INTER_AREA)
                    return cam_img
                w = int(gt_box[2]-gt_box[0])
                h = int(gt_box[3] - gt_box[1])
                cam1=vis_cam(cam,0,h,w,(123.7, 116.3, 103.5))
                cam2 = vis_cam(cam, 1,h,w,(123.7, 116.3, 140.5))
                cam3 = vis_cam(cam, 2,h,w,(123.7, 116.3, 103.5))
                mst=np.stack([mst*255]*3).transpose(1,2,0)

                dst_show=np.concatenate((dst,mst,cam1,cam2,cam3),axis=1)



                Image.fromarray(dst_show.astype(np.uint8)).save('/shared/ReferCam/cam_out/' + str(im_id[0].split(".")[0]) + ".jpg")


            pred_bbox[:,0], pred_bbox[:,2] = (pred_bbox[:,0]-dw)/ratio, (pred_bbox[:,2]-dw)/ratio
            # pred_bbox[:,1], pred_bbox[:,3] = (pred_bbox[:,1]-dh)/ratio, (pred_bbox[:,3]-dh)/ratio
            target_bbox[:,0], target_bbox[:,2] = (target_bbox[:,0]-dw)/ratio, (target_bbox[:,2]-dw)/ratio
            target_bbox[:,1], target_bbox[:,3] = (target_bbox[:,1]-dh)/ratio, (target_bbox[:,3]-dh)/ratio

            ## convert pred, gt box to original scale with meta-info
            top, bottom = round(float(dh[0]) - 0.1), args.size - round(float(dh[0]) + 0.1)
            left, right = round(float(dw[0]) - 0.1), args.size - round(float(dw[0]) + 0.1)
            img_np = imgs[0,:,top:bottom,left:right].data.cpu().numpy().transpose(1,2,0)


            ratio = float(ratio)
            new_shape = (round(img_np.shape[1] / ratio), round(img_np.shape[0] / ratio))
            ## also revert image for visualization
            img_np = cv2.resize(img_np, new_shape, interpolation=cv2.INTER_CUBIC)
            img_np = Variable(torch.from_numpy(img_np.transpose(2,0,1)).cuda().unsqueeze(0))

            pred_bbox[:,:2], pred_bbox[:,2], pred_bbox[:,3] = \
                torch.clamp(pred_bbox[:,:2], min=0), torch.clamp(pred_bbox[:,2], max=img_np.shape[3]), torch.clamp(pred_bbox[:,3], max=img_np.shape[2])
            target_bbox[:,:2], target_bbox[:,2], target_bbox[:,3] = \
                torch.clamp(target_bbox[:,:2], min=0), torch.clamp(target_bbox[:,2], max=img_np.shape[3]), torch.clamp(target_bbox[:,3], max=img_np.shape[2])

            n_feat=0
            b_eval=True
            if b_eval:
                # mask=mask.data.cpu().numpy()
                mask_np = mask[:,top:bottom, left:right].data.cpu().numpy().transpose(1, 2, 0)
                mask_np = cv2.resize(mask_np, new_shape, interpolation=cv2.INTER_CUBIC)

                predicts=np.zeros((new_shape[1],new_shape[0]))
                pred=pred_bbox.long().squeeze().numpy()

                cam_shape = (max(int(pred[3] - pred[1]),1), max(int(pred[2] - pred[0]),1))



                cam1=cam[n_feat]
                cam1=F.interpolate(cam1, cam_shape, mode='bilinear', align_corners=True).squeeze(0).data.cpu().numpy()
                cam1=np.argmax(cam1,0)


                predicts[int(pred[1]):int(pred[3]),int(pred[0]):int(pred[2])]=cam1
                # predicts[int(pred[1]):int(pred[3]), int(pred[0]):int(pred[2])]=1.
                mask_np=(mask_np>0.5).astype(np.float64)
                # predicts[int(pred[1]):int(pred[3]), int(pred[0]):int(pred[2])]=mask_np[int(pred[1]):int(pred[3]), int(pred[0]):int(pred[2])]


                # # sum_cam = np.sum(cam_list, axis=0)
                # sum_cam[sum_cam < 0] = 0
                # cam_max = np.max(sum_cam, (1, 2), keepdims=True)
                # cam_min = np.min(sum_cam, (1, 2), keepdims=True)
                # sum_cam[sum_cam < cam_min + 1e-5] = 0
                # norm_cam = (sum_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)




                I, U = compute_mask_IU(predicts, mask_np)
                IU_result.append({'batch_no': batch_idx, 'I': I, 'U': U})
                mean_IoU += float(I) / U
                cum_I += I
                cum_U += U
                msg = 'cumulative IoU = %f' % (cum_I/cum_U)
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (I/U >= eval_seg_iou)
                # print(msg)
                seg_total += 1
                tbar.set_description("Mean IoU: %.4f" % (mean_IoU/seg_total))



        iou = 0 #bbox_iou(pred_bbox, target_bbox.data.cpu(), x1y1x2y2=True)
        accu_center = compute_point_box(np.array(pred_bbox),bbox_gt.data.cpu().numpy())#compute_dists(np.array(pred_bbox),bbox.data.cpu().numpy(),5)/args.batch_size#np.sum(np.array((np.array(target_best_gi) == np.array(pred_gi)) * (np.array(target_best_gj) == np.array(pred_gj)), dtype=float))/args.batch_size
        accu = 0#np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        gt_onehot=[]#np.array((iou.data.cpu().numpy()>0.5),dtype=float)
        if args.seg:
            for ii in range(len(bi_score)):
                accr=np.sum(np.array(bi_score[ii].max(0)[1].data.cpu().numpy()== gt_onehot,dtype=float))
                acc_refine.update(accr, imgs.size(0))

        acc.update(accu, imgs.size(0))
        acc_center.update(accu_center, imgs.size(0))
        miou.update(0, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        tbar.set_description(
            'Acc in box {acc_refine.val:.4f} ({acc_refine.avg:.4f})'
            .format(acc_refine=acc_center))

        if args.save_plot:
            if batch_idx%1==0:
                save_segmentation_map(pred_bbox,target_bbox,img_np,'test',batch_idx*imgs.size(0),\
                    save_path='./visulizations/%s/'%args.dataset)
        
        if (batch_idx+1) % args.print_freq == 0 or (batch_idx+1)==len(val_loader):
            print_str = '[{0}/{1}] ' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) ' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f}) ' \
                'Accu in box {acc.val:.4f} ({acc.avg:.4f}) ' \
                .format( \
                    batch_idx+1, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc_center, acc_c=acc_center, miou=miou, acc_r=acc_refine)
            print(print_str)
            # print(msg)
            logging.info(print_str)

    if args.seg:
        print('Segmentation evaluation (without DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                          (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
        result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I / cum_U, mean_IoU / seg_total)
        print(result_str)
    # print(best_n_list, pred_best_n)
    # print(np.array(target_gi), np.array(pred_gi))
    # print(np.array(target_gj), np.array(pred_gj),'-')
    # print(acc.avg, miou.avg,acc_center.avg)

    # print_str = '[{0}/{1}]\t' \
    #     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
    #     'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
    #     'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
    #     'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
    #     'Accu_c {acc_c.val:.4f} ({acc_c.avg:.4f})\t' \
    #     .format( \
    #         batch_idx, len(val_loader), batch_time=batch_time, \
    #         data_time=data_time, \
    #         acc=acc, acc_c=acc_center, miou=miou)
    # print(print_str)

    logging.info("%f,%f,%f"%(acc.avg, float(miou.avg),acc_center.avg))
    return acc_center.avg


if __name__ == "__main__":
    main()
