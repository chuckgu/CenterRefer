import torch
import torch.nn as nn


class SegmentationLosses:
    def __init__(
        self,
        weight=None,
        size_average=True,
        batch_average=True,
        ignore_index=255,
        cuda=False,
    ):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode="ce"):
        """Choices: ['ce' or 'focal']"""
        if mode == "ce":
            return self.CrossEntropyLoss
        elif mode == "be":
            return self.BCEWithLogitsLoss
        elif mode == "focal":
            return self.FocalLoss
        elif mode == "ce_finetune":
            return self.CrossEntropyLossFinetune
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, _, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def BCEWithLogitsLoss(self, logit, target):
        n, _, h, w = logit.size()
        criterion = nn.BCEWithLogitsLoss()
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit.view(n,-1), target.view(n,-1))

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLossFinetune(self, logit, target):
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, size_average=self.size_average
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= logit.shape[0]

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, _, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class GMMNLoss:
    def __init__(self, sigma=[2, 5, 10, 20, 40, 80], cuda=False):
        self.sigma = sigma
        self.cuda = cuda

    def build_loss(self):
        return self.moment_loss

    def get_scale_matrix(self, M, N):
        s1 = torch.ones((N, 1)) * 1.0 / N
        s2 = torch.ones((M, 1)) * -1.0 / M
        if self.cuda:
            s1, s2 = s1.cuda(), s2.cuda()
        return torch.cat((s1, s2), 0)

    def moment_loss(self, gen_samples, x):
        X = torch.cat((gen_samples, x), 0)
        XX = torch.matmul(X, X.t())
        X2 = torch.sum(X * X, 1, keepdim=True)
        exp = XX - 0.5 * X2 - 0.5 * X2.t()
        M = gen_samples.size()[0]
        N = x.size()[0]
        s = self.get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        for v in self.sigma:
            kernel_val = torch.exp(exp / v)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)
        return loss


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.size(0)
        # num_joints = output.size(1)
        heatmap_pred = output.reshape((batch_size,  -1))
        heatmap_gt = target.reshape((batch_size,  -1))
        loss = 0

        if self.use_target_weight:
            loss += 0.5 * self.criterion(
                heatmap_pred.mul(target_weight[:]),
                heatmap_gt.mul(target_weight[:])
            )
        else:
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss