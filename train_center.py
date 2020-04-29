import os

import numpy as np
import torch
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.deeplab import DeepLab
from modeling.center_refer import CenterRefer
from modeling.sync_batchnorm.replicate import patch_replication_callback
from dataloaders.datasets import DATASETS_DIRS
from utils.calculate_weights import calculate_weigths_labels
from utils.loss import SegmentationLosses
from utils.loss import *
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from parsing import get_parser
from exp_data import CLASSES_NAMES
from base_trainer import BaseTrainer
from util import data_reader
from util.processing_tools import *
import torch.nn as nn
from util import im_processing, text_processing, eval_tools

class Trainer(BaseTrainer):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        b_test=None
        # Define Dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        (self.train_loader, self.val_loader, _, self.nclass,) = make_data_loader(
            args, b_test=b_test, **kwargs
        )
        self.nclass=1
        dataset=args.dataset
        setname='train'

        data_folder = './' + dataset + '/' + setname + '_batch/'
        data_prefix = dataset + '_' + setname


        # im_h, im_w, num_steps = model.H, model.W, model.num_steps
        # text_batch = np.zeros((bs, num_steps), dtype=np.float32)
        # image_batch = np.zeros((bs, im_h, im_w, 3), dtype=np.float32)
        # mask_batch = np.zeros((bs, im_h, im_w, 1), dtype=np.float32)

        # self.train_loader = data_reader.DataReader(data_folder, data_prefix,iters_per_log=100)

        # Define network
        base_model = DeepLab(
            num_classes=self.nclass,
            output_stride=16,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            pretrained=args.deeplab_pretrained,
            pretrained_path=args.deeplab_pretrained_path,
        )

        model = CenterRefer(vis_emb_net=base_model)

        train_backbone=True
        if train_backbone:

            train_params = [
                {"params": model.vis_emb_net.get_1x_lr_params(), "lr": args.lr},
                {"params": model.vis_emb_net.get_10x_lr_params(), "lr": args.lr },
                {"params": model.get_params(), "lr": args.lr*5 },
                {"params": model.get_params_slow(), "lr": args.lr*0.1},
            ]
        else:
            train_params = [
                {"params": model.get_params(), "lr": args.lr*5 },
                {"params": model.get_params_slow(), "lr": args.lr*0.1},
            ]

        # Define Optimizer
        # optimizer = torch.optim.SGD(
        #     train_params,
        #     momentum=args.momentum,
        #     weight_decay=args.weight_decay,
        #     nesterov=args.nesterov,
        # )
        optimizer = torch.optim.AdamW(
            train_params,
            # momentum=args.momentum,
            weight_decay=args.weight_decay,
            # nesterov=args.nesterov,
        )

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = (
                DATASETS_DIRS[args.dataset] / args.dataset + "_classes_weights.npy"
            )
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(
                    args.dataset, self.train_loader, self.nclass
                )
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.center_criterion = JointsMSELoss().cuda()
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(
            mode=args.loss_type
        )

        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_loader),1
        )

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            # if not os.path.isfile(args.resume):
            #     raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            if args.cuda:
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print("=> loaded checkpoint %s (epoch % .f)" % (args.resume,args.start_epoch))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def validation(self, epoch):
        class_names = ["RES"]
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        IU_result = list()
        score_thresh = 1e-9
        eval_seg_iou_list = [.5, .6, .7, .8, .9]
        cum_I, cum_U = 0, 0
        mean_IoU, mean_dcrf_IoU = 0, 0
        seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
        seg_total = 0.

        for i, sample in enumerate(tbar):
            image, target,text, heatmap_gt = sample["image"], sample["label"],  sample['text'],sample['center']
            if self.args.cuda:
                image, target,text,heatmap_gt = image.cuda(), target.cuda(),text.cuda(),heatmap_gt.cuda()
            with torch.no_grad():
                output,_ = self.model((image,text,heatmap_gt))
            loss = self.criterion(output, target)
            test_loss += loss.item()


            # pred = output.data.cpu().numpy()
            # predicts=nn.functional.sigmoid(output).cpu().numpy().squeeze(1)
            mask = target.cpu().numpy()
            predicts = (output.cpu().numpy().squeeze(1) >= score_thresh).astype(np.float32)
            # pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            # self.evaluator.add_batch(target, pred)
            # print(predicts.sum())
            n_iter=i
            I, U = eval_tools.compute_mask_IU(predicts, mask)
            IU_result.append({'batch_no': n_iter, 'I': I, 'U': U})
            mean_IoU += float(I) / U
            cum_I += I
            cum_U += U
            msg = 'cumulative IoU = %f' % (cum_I/cum_U)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I/U >= eval_seg_iou)
            # print(msg)
            seg_total += 1
            tbar.set_description("Test loss: %.3f,Mean IoU: %.4f" % (test_loss / (i + 1), mean_IoU/seg_total))

        print('Segmentation evaluation (without DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                          (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
        result_str += 'overall IoU = %f; mean IoU = %f\n' % (cum_I / cum_U, mean_IoU / seg_total)
        print(result_str)

        # Fast test during the training
        # Acc = self.evaluator.Pixel_Accuracy()
        # Acc_class, Acc_class_by_class = self.evaluator.Pixel_Accuracy_Class()
        # mIoU, mIoU_by_class = self.evaluator.Mean_Intersection_over_Union()
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # self.writer.add_scalar("val/total_loss_epoch", test_loss, epoch)
        # self.writer.add_scalar("val/mIoU", mean_IoU, epoch)
        # self.writer.add_scalar("val/Acc", Acc, epoch)
        # self.writer.add_scalar("val/Acc_class", Acc_class, epoch)
        # self.writer.add_scalar("val/fwIoU", FWIoU, epoch)
        # print("Validation:")
        # print(
        #     "[Epoch: %d, numImages: %5d]"
        #     % (epoch, i * self.args.batch_size + image.data.shape[0])
        # )
        # print("Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}")
        # print("Loss: {test_loss:.3f}")
        #
        # for i, (class_name, acc_value, mIoU_value) in enumerate(
        #     zip(class_names, Acc_class_by_class, mIoU_by_class)
        # ):
        #     self.writer.add_scalar("Acc_by_class/" + class_name, acc_value, epoch)
        #     self.writer.add_scalar("mIoU_by_class/" + class_name, mIoU_value, epoch)
        #     print(class_names[i], "- acc:", acc_value, " mIoU:", mIoU_value)

        new_pred = mean_IoU / seg_total
        is_best = False
        if new_pred>self.best_pred:
            is_best = True
            self.best_pred = new_pred
        self.saver.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
            },
            is_best,
        )


def main():
    parser = get_parser()
    parser.add_argument(
        "--deeplab_pretrained",
        type=bool,
        default=True,
        help="imagenet pretrained backbone",
    )
    # parser.add_argument('--workers', type=int, default=4,
    #                     metavar='N', help='dataloader threads')
    parser.add_argument(
        "--out-stride", type=int, default=16, help="network output stride (default: 8)"
    )

    # PASCAL VOC
    parser.add_argument(
        "--dataset",
        type=str,
        default="Gref",
        choices=["pascal", "coco", "cityscapes"],
        help="dataset name (default: pascal)",
    )

    parser.add_argument(
        "--use-sbd",
        action="store_true",
        default=False,
        help="whether to use SBD dataset (default: True)",
    )
    parser.add_argument("--base-size", type=int, default=320, help="base image size")
    parser.add_argument("--crop-size", type=int, default=320, help="crop image size")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="be",
        choices=["ce", "focal"],
        help="loss func type (default: ce)",
    )
    # training hyper params

    # PASCAL VOC
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )

    # PASCAL VOC
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: auto)",
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )

    parser.add_argument(
        "--deeplab_pretrained_path",
        type=str,
        default="checkpoint/deeplab-resnet.pth.tar",
        help="set the checkpoint name",
    )

    parser.add_argument(
        "--checkname",
        type=str,
        default="Gref_center",
        help="set the checkpoint name",
    )

    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=1, help="evaluation interval (default: 1)"
    )

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1",
        help="use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)",
    )
    # only seen classes
    # 10 unseen
    # parser.add_argument('--unseen_classes_idx', type=int, default=[10, 14, 1, 18, 8, 20, 19, 5, 9, 16])
    # 8 unseen
    # parser.add_argument('--unseen_classes_idx', type=int, default=[10, 14, 1, 18, 8, 20, 19, 5])
    # 6 unseen
    # parser.add_argument('--unseen_classes_idx', type=int, default=[10, 14, 1, 18, 8, 20])
    # 4 unseen
    # parser.add_argument('--unseen_classes_idx', type=int, default=[10, 14, 1, 18])
    # 2 unseen

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError(
                "Argument --gpu_ids must be a comma-separated list of integers only"
            )

    args.sync_bn = args.cuda and len(args.gpu_ids) > 1

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            "Gref": 30,
            "cityscapes": 200,
            "pascal": 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            "Gref": 0.1,
            "cityscapes": 0.01,
            "pascal": 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
    args.lr = 1E-3 / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = "deeplab-resnet-center"

    print(args)
    b_eval=False
    torch.manual_seed(args.seed)
    if b_eval:
        print("##################test mode#########################")
        args.resume="/shared/CenterRefer/run/Gref/Gref_center/experiment/"+"1_model.pth.tar"
        # args.resume="/home/tips/Desktop/project/CenterRefer/run/Gref/Gref_center/experiment/"+"13_model.pth.tar"
        trainer = Trainer(args)
        trainer.validation(0)
    else:
        print("###############train mode#######################")

        # args.resume = "/shared/CenterRefer/run/Gref/Gref_center/experiment/" + "1_model.pth.tar"
        trainer = Trainer(args)
        print("Starting Epoch:", trainer.args.start_epoch)
        print("Total Epoches:", trainer.args.epochs)

        # trainer.validation(trainer.args.start_epoch)
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val and epoch % args.eval_interval == (
                args.eval_interval - 1
            ):
                trainer.validation(epoch)
        trainer.writer.close()


if __name__ == "__main__":
    main()
