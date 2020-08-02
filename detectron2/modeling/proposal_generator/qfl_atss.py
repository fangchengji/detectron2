#!/usr/bin/env python3
# @Time    : 27/6/29 2:46 PM
# @Author  : fangcheng.ji
# @FileName: qfl_atss.py

import math
import torch
import torch.nn.functional as F
from torch import nn
import os
from typing import Dict, List

from .fcos import Scale
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec, cat, ml_nms, quality_focal_loss

from detectron2.layers import DFConv2d, get_norm
from detectron2.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou
from detectron2.utils.comm import get_world_size, reduce_sum
from fvcore.nn import sigmoid_focal_loss_jit

from ..anchor_generator import build_anchor_generator
from ..matcher import Matcher

from ..backbone.bifpn import SeparableConvBlock
from ..backbone.efficientnet import MemoryEfficientSwish, Swish


INF = 100000000


@PROPOSAL_GENERATOR_REGISTRY.register()
class QFLATSS(torch.nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(QFLATSS, self).__init__()
        self.cfg = cfg

        self.in_features = cfg.MODEL.ATSS.IN_FEATURES

        feature_shapes = [input_shape[f] for f in self.in_features]
        in_channels = [f.channels for f in feature_shapes]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        if cfg.MODEL.ATSS.HEAD == "dw":
            self.fcos_head = DWATSSHead(cfg, in_channels)
        else:
            self.fcos_head = ATSSHead(cfg, in_channels)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = ATSSLossComputation(cfg, box_coder)
        # for inference
        self.box_selector_test = ATSSPostProcessor(
            pre_nms_thresh=cfg.MODEL.ATSS.INFERENCE_TH,
            pre_nms_top_n=cfg.MODEL.ATSS.PRE_NMS_TOP_N,
            nms_thresh=cfg.MODEL.ATSS.NMS_TH,
            fpn_post_nms_top_n=cfg.MODEL.ATSS.POST_NMS_TOPK_TEST,
            min_size=0,
            num_classes=cfg.MODEL.ATSS.NUM_CLASSES + 1,  # add background
            bbox_aug_enabled=cfg.TEST.AUG.ENABLED,
            box_coder=box_coder,
        )
        # self.anchor_generator = make_anchor_generator_atss(cfg)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

    def forward(self, images, features, gt_instances):
        features = [features[f] for f in self.in_features]
        box_cls, box_regression = self.fcos_head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            return self._forward_train(box_cls, box_regression, gt_instances, anchors)
        else:
            return self._forward_test(images.image_sizes, box_cls, box_regression, anchors)

    def _forward_train(self, box_cls, box_regression, gt_instances, anchors):
        loss_box_cls, loss_box_reg = self.loss_evaluator(
            box_cls, box_regression, gt_instances, anchors
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
        }
        return None, losses

    def _forward_test(self, image_sizes, box_cls, box_regression, anchors):
        boxes = self.box_selector_test(image_sizes, box_cls, box_regression, anchors)
        return boxes, {}


class ATSSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ATSSHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ATSS.NUM_CLASSES
        num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]) * len(cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])

        head_configs = {"cls": (cfg.MODEL.ATSS.NUM_CONVS,
                                False),
                        "bbox": (cfg.MODEL.ATSS.NUM_CONVS,
                                 cfg.MODEL.ATSS.USE_DCN_IN_TOWER),
                        }
        norm = None if cfg.MODEL.ATSS.NORM == "none" else cfg.MODEL.ATSS.NORM

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            if use_deformable:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            for i in range(num_convs):
                tower.append(conv_func(
                        in_channels, in_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, in_channels))
                elif norm is not None:
                    tower.append(get_norm(norm, in_channels))

                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            assert num_anchors == 1, "regressing from a point only support num_anchors == 1"
            torch.nn.init.constant_(self.bbox_pred.bias, 4)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
                bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

        return logits, bbox_reg


class DWATSSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(DWATSSHead, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.ATSS.NUM_CLASSES
        num_anchors = len(cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]) * len(cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])

        head_configs = {"cls": (cfg.MODEL.ATSS.NUM_CONVS,
                                False),
                        "bbox": (cfg.MODEL.ATSS.NUM_CONVS,
                                 cfg.MODEL.ATSS.USE_DCN_IN_TOWER),
                        }
        
        # self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.swish = MemoryEfficientSwish()

        norm = None if cfg.MODEL.ATSS.NORM == "none" else cfg.MODEL.ATSS.NORM

        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            for i in range(num_convs):
                tower.append(
                    SeparableConvBlock(in_channels, in_channels, activation=False)
                )
                tower.append(
                    get_norm(norm, in_channels, momentum=0.01, eps=1e-3)
                )
                tower.append(self.swish)
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cls_logits = SeparableConvBlock(
            in_channels, num_anchors * num_classes, activation=False
        )
        self.bbox_pred = SeparableConvBlock(
            in_channels, num_anchors * 4, activation=False
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if hasattr(l, "bias") and l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        # prior_prob = cfg.MODEL.ATSS.PRIOR_PROB
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # torch.nn.init.constant_(self.cls_logits.pointwise_conv.conv.bias, bias_value)
        # if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
        #     assert num_anchors == 1, "regressing from a point only support num_anchors == 1"
        #     torch.nn.init.constant_(self.bbox_pred.bias, 4)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
                bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

        return logits, bbox_reg


class ATSSLossComputation(object):

    def __init__(self, cfg, box_coder):
        self.cfg = cfg
        self.focal_loss_alpha = cfg.MODEL.ATSS.LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.ATSS.LOSS_GAMMA
        self.num_classes = cfg.MODEL.ATSS.NUM_CLASSES

        self.matcher = Matcher(
            cfg.MODEL.ATSS.IOU_THRESHOLDS,
            cfg.MODEL.ATSS.IOU_LABELS,
            allow_low_quality_matches=True
        )
        self.box_coder = box_coder

    def GIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def DIoULoss(self, pred, target, anchor, weight=None):
        pred_boxes = self.box_coder.decode(pred.view(-1, 4), anchor.view(-1, 4))
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        pred_cx = (pred_x2 + pred_x1) / 2.0
        pred_cy = (pred_y2 + pred_y1) / 2.0

        gt_boxes = self.box_coder.decode(target.view(-1, 4), anchor.view(-1, 4))
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        target_cx = (target_x2 + target_x1) / 2.0
        target_cy = (target_y2 + target_y1) / 2.0

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)

        c_squared = torch.pow(y2_enclosing - y1_enclosing, 2) + torch.pow(x2_enclosing - x1_enclosing, 2) + 1e-7
        d_squared = torch.pow(target_cy - pred_cy, 2) + torch.pow(target_cx - pred_cx, 2)

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        dious = ious - d_squared / c_squared
        losses = 1 - dious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

    def prepare_targets(self, gt_instances, anchors):
        cls_labels = []
        reg_targets = []
        anchors_all_level = Boxes.cat(anchors)
        for im_i in range(len(gt_instances)):
            targets_per_im = gt_instances[im_i]
            bboxes_per_im = targets_per_im.gt_boxes
            labels_per_im = targets_per_im.gt_classes
            num_gt = len(bboxes_per_im)

            if num_gt > 0:
                if self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'SSC':
                    object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
                    area_per_im = targets_per_im.area()
                    expanded_object_sizes_of_interest = []
                    points = []
                    for l, anchors_per_level in enumerate(anchors[im_i]):
                        anchors_per_level = anchors_per_level.bbox
                        anchors_cx_per_level = (anchors_per_level[:, 2] + anchors_per_level[:, 0]) / 2.0
                        anchors_cy_per_level = (anchors_per_level[:, 3] + anchors_per_level[:, 1]) / 2.0
                        points_per_level = torch.stack((anchors_cx_per_level, anchors_cy_per_level), dim=1)
                        points.append(points_per_level)
                        object_sizes_of_interest_per_level = \
                            points_per_level.new_tensor(object_sizes_of_interest[l])
                        expanded_object_sizes_of_interest.append(
                            object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
                        )
                    expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
                    points = torch.cat(points, dim=0)

                    xs, ys = points[:, 0], points[:, 1]
                    l = xs[:, None] - bboxes_per_im[:, 0][None]
                    t = ys[:, None] - bboxes_per_im[:, 1][None]
                    r = bboxes_per_im[:, 2][None] - xs[:, None]
                    b = bboxes_per_im[:, 3][None] - ys[:, None]
                    reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

                    is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0.01

                    max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
                    is_cared_in_the_level = \
                        (max_reg_targets_per_im >= expanded_object_sizes_of_interest[:, [0]]) & \
                        (max_reg_targets_per_im <= expanded_object_sizes_of_interest[:, [1]])

                    locations_to_gt_area = area_per_im[None].repeat(len(points), 1)
                    locations_to_gt_area[is_in_boxes == 0] = INF
                    locations_to_gt_area[is_cared_in_the_level == 0] = INF
                    locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

                    cls_labels_per_im = labels_per_im[locations_to_gt_inds]
                    cls_labels_per_im[locations_to_min_area == INF] = self.num_classes
                    matched_gts = bboxes_per_im[locations_to_gt_inds]
                elif self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'ATSS':
                    num_anchors_per_loc = len(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS[0]) \
                                          * len(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0])

                    # TODO deal with num gt is 0
                    num_anchors_per_level = [len(anchors_per_level) for anchors_per_level in anchors]
                    ious = pairwise_iou(anchors_all_level, bboxes_per_im)

                    gt_cx = (bboxes_per_im.tensor[:, 2] + bboxes_per_im.tensor[:, 0]) / 2.0
                    gt_cy = (bboxes_per_im.tensor[:, 3] + bboxes_per_im.tensor[:, 1]) / 2.0
                    gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                    anchors_cx_per_im = (anchors_all_level.tensor[:, 2] + anchors_all_level.tensor[:, 0]) / 2.0
                    anchors_cy_per_im = (anchors_all_level.tensor[:, 3] + anchors_all_level.tensor[:, 1]) / 2.0
                    anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                    distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

                    # Selecting candidates based on the center distance between anchor box and object
                    candidate_idxs = []
                    star_idx = 0
                    for level, anchors_per_level in enumerate(anchors):
                        end_idx = star_idx + num_anchors_per_level[level]
                        distances_per_level = distances[star_idx:end_idx, :]
                        topk = min(self.cfg.MODEL.ATSS.TOPK * num_anchors_per_loc, num_anchors_per_level[level])
                        _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                        candidate_idxs.append(topk_idxs_per_level + star_idx)
                        star_idx = end_idx
                    candidate_idxs = torch.cat(candidate_idxs, dim=0)

                    # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
                    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
                    iou_mean_per_gt = candidate_ious.mean(0)
                    iou_std_per_gt = candidate_ious.std(0)
                    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
                    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

                    # Limiting the final positive samples’ center to object
                    anchor_num = anchors_cx_per_im.shape[0]
                    for ng in range(num_gt):
                        candidate_idxs[:, ng] += ng * anchor_num
                    e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                    e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
                    candidate_idxs = candidate_idxs.view(-1)
                    l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bboxes_per_im.tensor[:, 0]
                    t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im.tensor[:, 1]
                    r = bboxes_per_im.tensor[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
                    b = bboxes_per_im.tensor[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
                    is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                    is_pos = is_pos & is_in_gts

                    # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
                    ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
                    index = candidate_idxs.view(-1)[is_pos.view(-1)]
                    ious_inf[index] = ious.t().contiguous().view(-1)[index]
                    ious_inf = ious_inf.view(num_gt, -1).t()

                    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
                    cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                    cls_labels_per_im[anchors_to_gt_values == -INF] = self.num_classes
                    matched_gts = bboxes_per_im[anchors_to_gt_indexs]
                elif self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'TOPK':
                    gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
                    gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
                    gt_points = torch.stack((gt_cx, gt_cy), dim=1)

                    anchors_cx_per_im = (anchors_all_level.tensor[:, 2] + anchors_all_level.tensor[:, 0]) / 2.0
                    anchors_cy_per_im = (anchors_all_level.tensor[:, 3] + anchors_all_level.tensor[:, 1]) / 2.0
                    anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

                    distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
                    distances = distances / distances.max() / 1000
                    ious = pairwise_iou(anchors_all_level, bboxes_per_im)

                    is_pos = ious * False
                    for ng in range(num_gt):
                        _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(self.cfg.MODEL.ATSS.TOPK, dim=0)
                        l = anchors_cx_per_im[topk_idxs] - bboxes_per_im[ng, 0]
                        t = anchors_cy_per_im[topk_idxs] - bboxes_per_im[ng, 1]
                        r = bboxes_per_im[ng, 2] - anchors_cx_per_im[topk_idxs]
                        b = bboxes_per_im[ng, 3] - anchors_cy_per_im[topk_idxs]
                        is_in_gt = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                        is_pos[topk_idxs[is_in_gt == 1], ng] = True

                    ious[is_pos == 0] = -INF
                    anchors_to_gt_values, anchors_to_gt_indexs = ious.max(dim=1)

                    cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
                    cls_labels_per_im[anchors_to_gt_values == -INF] = self.num_classes
                    matched_gts = bboxes_per_im[anchors_to_gt_indexs]
                elif self.cfg.MODEL.ATSS.POSITIVE_TYPE == 'IoU':
                    match_quality_matrix = pairwise_iou(bboxes_per_im, anchors_all_level)
                    matched_idxs = self.matcher(match_quality_matrix)
                    targets_per_im = targets_per_im.copy_with_fields(['labels'])
                    matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

                    cls_labels_per_im = matched_targets.get_field("labels")
                    cls_labels_per_im = cls_labels_per_im.to(dtype=torch.float32)

                    # Background (negative examples)
                    bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                    cls_labels_per_im[bg_indices] = 0

                    # discard indices that are between thresholds
                    inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                    cls_labels_per_im[inds_to_discard] = -1

                    matched_gts = matched_targets.bbox

                    # Limiting positive samples’ center to object
                    # in order to filter out poor positives and use the centerness branch
                    pos_idxs = torch.nonzero(cls_labels_per_im > 0).squeeze(1)
                    pos_anchors_cx = (anchors_all_level.tensor[pos_idxs, 2] + anchors_all_level.tensor[pos_idxs, 0]) / 2.0
                    pos_anchors_cy = (anchors_all_level.tensor[pos_idxs, 3] + anchors_all_level.tensor[pos_idxs, 1]) / 2.0
                    l = pos_anchors_cx - matched_gts[pos_idxs, 0]
                    t = pos_anchors_cy - matched_gts[pos_idxs, 1]
                    r = matched_gts[pos_idxs, 2] - pos_anchors_cx
                    b = matched_gts[pos_idxs, 3] - pos_anchors_cy
                    is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                    cls_labels_per_im[pos_idxs[is_in_gts == 0]] = -1
                else:
                    raise NotImplementedError

                reg_targets_per_im = self.box_coder.encode(matched_gts.tensor, anchors_all_level.tensor)

            else:   # no gt instance
                # all negative
                reg_targets_per_im = torch.zeros_like(anchors_all_level.tensor)
                cls_labels_per_im = torch.zeros(
                    len(anchors_all_level.tensor),
                    dtype=torch.long,
                    device=anchors_all_level.device
                ) + self.num_classes

            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
        return cls_labels, reg_targets

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def compute_iou_score(self, reg_preds, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        preds = self.box_coder.decode(reg_preds, anchors)
        gts = Boxes(gts)
        preds = Boxes(preds)
        return matched_boxlist_iou(preds, gts)

    def __call__(self, box_cls, box_regression, gt_instances, anchors):
        labels, reg_targets = self.prepare_targets(gt_instances, anchors)

        N = len(labels)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(box_cls, box_regression)

        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        anchors_flatten = torch.cat([Boxes.cat(anchors).tensor for _ in range(N)], dim=0)

        pos_inds = torch.nonzero(labels_flatten != self.num_classes).squeeze(1)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # one hot label for focal loss
        class_target = torch.zeros_like(box_cls_flatten)
        class_target[pos_inds, labels_flatten[pos_inds]] = 1

        # cls_loss = sigmoid_focal_loss_jit(
        #     box_cls_flatten,
        #     class_target,
        #     alpha=self.focal_loss_alpha,
        #     gamma=self.focal_loss_gamma,
        #     reduction="sum"
        # ) / num_pos_avg_per_gpu

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        anchors_flatten = anchors_flatten[pos_inds]
        centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
        sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

        # qfl score
        score = torch.zeros(class_target.size()[0], dtype=torch.float32, device=class_target.device)
        score[pos_inds] = self.compute_iou_score(
            box_regression_flatten.detach(),
            reg_targets_flatten,
            anchors_flatten
        )

        cls_loss = quality_focal_loss(
            box_cls_flatten,
            class_target,
            score,                # IoU score
            weight=1.0,           # weight = 1.0
            beta=self.focal_loss_gamma,
            reduction='mean',
            avg_factor=num_pos_avg_per_gpu,
        )

        if pos_inds.numel() > 0:
            reg_loss = self.DIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()

        return cls_loss, reg_loss * self.cfg.MODEL.ATSS.REG_LOSS_WEIGHT


class ATSSPostProcessor(torch.nn.Module):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        box_coder,
        bbox_aug_enabled=False,
    ):
        super(ATSSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.box_coder = box_coder

    def forward_for_single_feature_map(self, box_cls, box_regression, anchors):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds):

            per_box_cls = per_box_cls[per_candidate_inds]

            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                anchors.tensor[per_box_loc, :].view(-1, 4)
            )

            pred_boxes = Boxes(detections)
            scores = torch.sqrt(per_box_cls)
            pred_classes = per_class

            results.append((pred_boxes, scores, pred_classes))

        return results

    def forward(self, image_sizes, box_cls, box_regression, anchors):
        sampled_boxes = []
        # anchors = list(zip(*anchors))
        for _, (o, b, a) in enumerate(zip(box_cls, box_regression, anchors)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(o, b, a)
            )

        boxlists = []
        for i, image_size in enumerate(image_sizes):
            boxlist = Instances(image_size)
            boxes = []
            scores = []
            classes = []
            for j in range(len(anchors)):
                boxes.append(sampled_boxes[j][i][0])
                scores.append(sampled_boxes[j][i][1])
                classes.append(sampled_boxes[j][i][2])
            boxes = Boxes.cat(boxes)
            boxes.clip(image_size)
            keep = boxes.nonempty(self.min_size)
            boxlist.pred_boxes = boxes[keep]
            boxlist.scores = torch.cat(scores, dim=0)[keep]
            boxlist.pred_classes = torch.cat(classes, dim=0)[keep]

            boxlists.append(boxlist)

        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0][0] / self.cfg.MODEL.ATSS.FPN_STRIDES[0]
            l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
            t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
            r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
            b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
            targets = torch.stack([l, t, r, b], dim=1)
        elif self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            TO_REMOVE = 1  # TODO remove
            ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
            gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
            gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
            gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
            targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
            targets_dw = ww * torch.log(gt_widths / ex_widths)
            targets_dh = wh * torch.log(gt_heights / ex_heights)
            targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        if self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'POINT':
            TO_REMOVE = 1  # TODO remove
            anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
            anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

            w = self.cfg.MODEL.ANCHOR_GENERATOR.SIZES[0][0] / self.cfg.MODEL.ATSS.FPN_STRIDES[0]
            x1 = anchors_cx - preds[:, 0] / w * anchors_w
            y1 = anchors_cy - preds[:, 1] / w * anchors_h
            x2 = anchors_cx + preds[:, 2] / w * anchors_w
            y2 = anchors_cy + preds[:, 3] / w * anchors_h
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        elif self.cfg.MODEL.ATSS.REGRESSION_TYPE == 'BOX':
            anchors = anchors.to(preds.dtype)

            TO_REMOVE = 1  # TODO remove
            widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
            heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
            ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
            ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

            wx, wy, ww, wh = (10., 10., 5., 5.)
            dx = preds[:, 0::4] / wx
            dy = preds[:, 1::4] / wy
            dw = preds[:, 2::4] / ww
            dh = preds[:, 3::4] / wh

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=math.log(1000. / 16))
            dh = torch.clamp(dh, max=math.log(1000. / 16))

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(preds)
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


# def make_anchor_generator_atss(config):
#     anchor_sizes = config.MODEL.ATSS.ANCHOR_SIZES
#     aspect_ratios = config.MODEL.ATSS.ASPECT_RATIOS
#     anchor_strides = config.MODEL.ATSS.ANCHOR_STRIDES
#     straddle_thresh = config.MODEL.ATSS.STRADDLE_THRESH
#     octave = config.MODEL.ATSS.OCTAVE
#     scales_per_octave = config.MODEL.ATSS.SCALES_PER_OCTAVE
#
#     assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
#     new_anchor_sizes = []
#     for size in anchor_sizes:
#         per_layer_anchor_sizes = []
#         for scale_per_octave in range(scales_per_octave):
#             octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
#             per_layer_anchor_sizes.append(octave_scale * size)
#         new_anchor_sizes.append(tuple(per_layer_anchor_sizes))
#
#     anchor_generator = DefaultAnchorGenerator(
#         {
#             "sizes": new_anchor_sizes,
#             "aspect_ratios": aspect_ratios,
#             "strides": anchor_strides,
#             "offset": 0.0
#         }
#     )
#     return anchor_generator
