import torch
from torch import nn
import math


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        g_w_intersect = torch.max(pred_left, target_left) + \
                        torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                        torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        elif self.loc_loss_type == 'diou':
            rou_square = 0.25 * (torch.pow(target_right - pred_right + pred_left - target_left, 2) + \
                                 torch.pow(target_bottom - pred_bottom + pred_top - target_top, 2))
            c_square = torch.pow(g_w_intersect, 2) + torch.pow(g_h_intersect, 2)
            diou = ious - rou_square/(c_square + 0.00001)
            losses = 1 - diou
        elif self.loc_loss_type == 'ciou':
            rou_square = 0.25 * (torch.pow(target_right - pred_right + pred_left - target_left, 2) + \
                                 torch.pow(target_bottom - pred_bottom + pred_top - target_top, 2))
            c_square = torch.pow(g_w_intersect, 2) + torch.pow(g_h_intersect, 2)
            diou_loss = 1.0 - ious + rou_square / (c_square + 0.00001)
            # ciou
            w_gt = target_left + target_right
            h_gt = target_top + target_bottom + 0.00001
            w_pred = pred_left + pred_right
            h_pred = pred_top + pred_bottom + 0.00001
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
            with torch.no_grad():
                s = 1.0 - ious
                alpha = v / (s + v + 0.00001)
            losses = diou_loss + alpha * v
        else:
            raise NotImplementedError

        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()
