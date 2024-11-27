import random

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from isegm.utils import misc

class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)



class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

class ErrorCount(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt: torch.Tensor, label: torch.Tensor):
        size = np.prod(gt.size())
        diff_map = gt != (torch.sigmoid(label) > 0.49)
        diff_map = diff_map.sum() / size
        return diff_map

class PatchLoss(nn.Module):
    def __init__(self, rank=0):
        super(PatchLoss, self).__init__()
        self.loss_fn = TripletLoss()

    def forward(self, f_q, f_k, tau=0.07):
        B, _, H, W = f_k.shape
        total_loss = 0.0
        total_times = 0  # Number of times we compute the loss

        # Process all blocks to reduce variance
        for block in range(len(f_q)):
            sel_tokens_s = f_q[block]['sel_tokens_s']  # [B, T_s, C]
            idx_tokens_s = f_q[block]['idx_tokens_s']  # [B, T_s]
            sel_tokens_l = f_q[block]['sel_tokens_l']  # [B, T_l, C]
            idx_tokens_l = f_q[block]['idx_tokens_l']  # [B, T_l]
            kernel_s = f_q[block]['kernel_s']  # [B, C, 1]
            kernel_l = f_q[block]['kernel_l']  # [B, C, 1]

            # Remove zero tokens (entirely zero across batch and feature dimensions)
            mask_s = sel_tokens_s.abs().sum(dim=(0, 2)) != 0  # [T_s]
            sel_tokens_s = sel_tokens_s[:, mask_s, :]  # [B, T_s', C]
            idx_tokens_s = idx_tokens_s[:, mask_s]  # [B, T_s']

            mask_l = sel_tokens_l.abs().sum(dim=(0, 2)) != 0  # [T_l]
            sel_tokens_l = sel_tokens_l[:, mask_l, :]  # [B, T_l', C]
            idx_tokens_l = idx_tokens_l[:, mask_l]  # [B, T_l']

            # Interpolate f_k to required scales
            f_k_resized_s = F.interpolate(f_k, scale_factor=1 / 8, mode='nearest')  # [B, C, H_s, W_s]
            f_k_resized_l = F.interpolate(f_k, scale_factor=1 / 28, mode='nearest')  # [B, C, H_l, W_l]

            # Create valid masks
            valid_mask_s = f_k_resized_s.sum(dim=1) > 0  # [B, H_s, W_s]
            valid_mask_l = f_k_resized_l.sum(dim=1) > 0  # [B, H_l, W_l]

            # Flatten valid masks and indices
            valid_idx_s = valid_mask_s.view(B, -1)  # [B, N_s]
            valid_idx_l = valid_mask_l.view(B, -1)  # [B, N_l]

            # Process batch samples
            for i in range(B):
                # Small tokens
                idx_gt_s = valid_idx_s[i].nonzero(as_tuple=False).squeeze(-1)
                idx_token_s = idx_tokens_s[i]
                common_indices_s = torch.tensor(list(set(idx_gt_s.cpu().numpy()) & set(idx_token_s.cpu().numpy())),
                                                device=f_k.device)

                if common_indices_s.numel() == 0 or common_indices_s.numel() == idx_token_s.numel():
                    continue  # Skip if no valid triplets can be formed

                # Create a mask for positive and negative tokens
                pos_mask_s = torch.isin(idx_token_s, common_indices_s)
                neg_mask_s = ~pos_mask_s

                # Ensure there are positives and negatives
                if pos_mask_s.sum() == 0 or neg_mask_s.sum() == 0:
                    continue

                # Extract tokens
                pos_tokens_s = sel_tokens_s[i, pos_mask_s, :]  # [num_pos, C]
                neg_tokens_s = sel_tokens_s[i, neg_mask_s, :]  # [num_neg, C]
                anchor_s = kernel_s[i].squeeze(-1)  # [C]

                # Compute triplet loss for small tokens
                loss_s = self.loss_fn(anchor_s.unsqueeze(0), pos_tokens_s, neg_tokens_s)
                total_loss += loss_s
                total_times += 1

                # Large tokens
                idx_gt_l = valid_idx_l[i].nonzero(as_tuple=False).squeeze(-1)
                idx_token_l = idx_tokens_l[i]
                common_indices_l = torch.tensor(list(set(idx_gt_l.cpu().numpy()) & set(idx_token_l.cpu().numpy())),
                                                device=f_k.device)

                if common_indices_l.numel() == 0 or common_indices_l.numel() == idx_token_l.numel():
                    continue  # Skip if no valid triplets can be formed

                # Create a mask for positive and negative tokens
                pos_mask_l = torch.isin(idx_token_l, common_indices_l)
                neg_mask_l = ~pos_mask_l

                if pos_mask_l.sum() == 0 or neg_mask_l.sum() == 0:
                    continue

                # Extract tokens
                pos_tokens_l = sel_tokens_l[i, pos_mask_l, :]  # [num_pos, C]
                neg_tokens_l = sel_tokens_l[i, neg_mask_l, :]  # [num_neg, C]
                anchor_l = kernel_l[i].squeeze(-1)  # [C]

                # Compute triplet loss for large tokens
                loss_l = self.loss_fn(anchor_l.unsqueeze(0), pos_tokens_l, neg_tokens_l)
                total_loss += loss_l
                total_times += 1

        if total_times == 0:
            # No loss computed, return zero with requires_grad
            total_loss = torch.tensor(0.0, device=f_k.device, requires_grad=True)
        else:
            total_loss = total_loss / total_times

        return total_loss

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.to(anchor.device)
            ap_dist = torch.norm(anchor-pos, 2, dim=1).mean().view(-1)
            # ap_dist = F.cosine_similarity(anchor, pos).mean().unsqueeze(0)
            an_dist = torch.norm(anchor-neg, 2, dim=1).mean().view(-1)
            # an_dist = F.cosine_similarity(anchor, neg).mean().unsqueeze(0)

            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

class ScoreLoss(nn.Module):
    def __init__(self):
        super(ScoreLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, f_q, gt):
        B, _, _, _ = gt.shape
        loss = 0.0
        gt8 = F.interpolate(gt, scale_factor=1 / 8).view(B, -1)
        gt28 = F.interpolate(gt, scale_factor=1 / 32).view(B, -1)
        for block in range(len(f_q)):
            mask_s = ~(f_q[block]['pos_scores_s'].sum(dim=1) == 0)
            mask_l = ~(f_q[block]['pos_scores_l'].sum(dim=1) == 0)
            loss = loss + 0.5 * self.loss(f_q[block]['pos_scores_s'][mask_s], gt8[mask_s])
            loss = loss + 0.5 * self.loss(f_q[block]['pos_scores_l'][mask_l], gt28[mask_l])

        return loss / len(f_q)

    def log_states(self, sw, name, global_step):
        pass