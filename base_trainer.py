from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

mu = np.array((104.00698793, 116.66876762, 122.67891434))


def binary_cross_inbal(output, target):
    n, _, h, w = output.size()

    n_class=(target>0).sum()
    b_class=(n*h*w)-n_class

    pos_weight = torch.Tensor([int(b_class/n_class)]).cuda()
    loss = F.binary_cross_entropy_with_logits(output.view(n,-1), target.view(n,-1), pos_weight=pos_weight)

    loss /= n
    return loss

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class BaseTrainer:
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            if len(sample["image"]) >= 1:
                image, target = sample["image"], sample["label"]
                text = sample['text']
                heatmap_gt = sample['center']

                if self.args.cuda:
                    image, target,text,heatmap_gt = image.cuda(), target.cuda(),text.cuda(),heatmap_gt.cuda()
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.optimizer.zero_grad()
                output,center_mask = self.model((image,text,heatmap_gt))

                # mask_loss = self.criterion(output, target)

                mask_loss = binary_cross_inbal(output, target)
                # center_loss=self.center_criterion(center_mask,heatmap_gt)
                center_loss=_neg_loss(center_mask.squeeze(1),heatmap_gt)
                sigma=0.01
                loss=mask_loss+sigma*center_loss
                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ')
                    exit(1)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description("Train loss: %.4f, Mask:: %.4f, Center: %.4f" % (train_loss / (i + 1),mask_loss,center_loss))
                self.writer.add_scalar(
                    "train/total_loss_iter", loss.item(), i + num_img_tr * epoch
                )

                # # Show 10 * 3 inference results each epoch
                # if i % (num_img_tr // 10) == 0:
                #     global_step = i + num_img_tr * epoch
                #     self.summary.visualize_image(
                #         self.writer,
                #         self.args.dataset,
                #         image,
                #         target,
                #         output,
                #         global_step,
                #     )

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, i * self.args.batch_size + image.data.shape[0])
        )
        print("Loss: {train_loss:.3f}")

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
            )
