from tqdm import tqdm
import numpy as np
import torch

mu = np.array((104.00698793, 116.66876762, 122.67891434))

class BaseTrainer:
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            if len(sample["image"]) >= 1:
                image, target = sample["image"], sample["label"]
                # text = sample['text_batch']
                # image = sample['im_batch'].astype(np.float32)
                # target = torch.as_tensor(np.expand_dims(sample['mask_batch'].astype(np.float32), axis=2))
                #
                # img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
                # img_tmp *= (0.229, 0.224, 0.225)
                # img_tmp += (0.485, 0.456, 0.406)
                # img_tmp *= 255.0
                # img_tmp = img_tmp.astype(np.uint8)
                #
                # image=torch.as_tensor(np.expand_dims(image, axis=0))
                # image.shape=[b,3,513,513]
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description("Train loss: %.3f" % (train_loss / (i + 1)))
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
