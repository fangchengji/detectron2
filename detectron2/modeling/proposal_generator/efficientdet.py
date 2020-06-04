#!/usr/bin/env python3
# @Time    : 16/5/20 3:38 PM
# @Author  : fangcheng.ji
# @FileName: efficientdet.py

import logging
import torch
from torch import nn
import torch.nn.functional as F

from typing import Dict, List
import math
import copy

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.backbone.bifpn import SeparableConvBlock

from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.retinanet import \
    RetinaNet, permute_to_N_HWA_K


@PROPOSAL_GENERATOR_REGISTRY.register()
class EfficientDet(RetinaNet):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg)
        del self.backbone

        self.in_features = cfg.MODEL.EFFICIENTDET.IN_FEATURES
        self.num_classes = cfg.MODEL.EFFICIENTDET.NUM_CLASSES

        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.EFFICIENTDET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.EFFICIENTDET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.EFFICIENTDET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.EFFICIENTDET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.EFFICIENTDET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.EFFICIENTDET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period = cfg.VIS_PERIOD

        feature_shapes = [input_shape[f] for f in self.in_features]

        self.head = EfficientDetHead(cfg, feature_shapes)
        # self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        # self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.EFFICIENTDET.BBOX_REG_WEIGHTS)
        # self.matcher = Matcher(
        #     cfg.MODEL.EFFICIENTDET.IOU_THRESHOLDS,
        #     cfg.MODEL.EFFICIENTDET.IOU_LABELS,
        #     allow_low_quality_matches=True,
        # )

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        # self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        # self.loss_normalizer_momentum = 0.9

    def forward(self, images, features, gt_instances):
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
            losses = self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)
            return None, losses
        else:
            results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
            return results, {}

    # def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
    #     """
    #     Args:
    #         For `gt_classes` and `gt_anchors_deltas` parameters, see
    #             :meth:`RetinaNet.get_ground_truth`.
    #         Their shapes are (N, R) and (N, R, 4), respectively, where R is
    #         the total number of anchors across levels, i.e. sum(Hi x Wi x A)
    #         For `pred_class_logits` and `pred_anchor_deltas`, see
    #             :meth:`RetinaNetHead.forward`.
    #
    #     Returns:
    #         dict[str: Tensor]:
    #             mapping from a named loss to a scalar tensor
    #             storing the loss. Used during training only. The dict keys are:
    #             "loss_cls" and "loss_box_reg"
    #     """
    #     pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
    #         pred_class_logits, pred_anchor_deltas, self.num_classes
    #     )  # Shapes: (N x R, K) and (N x R, 4), respectively.
    #
    #     gt_classes = gt_classes.flatten()
    #     gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)
    #
    #     valid_idxs = gt_classes >= 0
    #     foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
    #     num_foreground = foreground_idxs.sum().item()
    #     get_event_storage().put_scalar("num_foreground", num_foreground)
    #     self.loss_normalizer = (
    #         self.loss_normalizer_momentum * self.loss_normalizer
    #         + (1 - self.loss_normalizer_momentum) * num_foreground
    #     )
    #
    #     gt_classes_target = torch.zeros_like(pred_class_logits)
    #     gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
    #
    #     # logits loss
    #     loss_cls = sigmoid_focal_loss_jit(
    #         pred_class_logits[valid_idxs],
    #         gt_classes_target[valid_idxs],
    #         alpha=self.focal_loss_alpha,
    #         gamma=self.focal_loss_gamma,
    #         reduction="sum",
    #     ) / max(1, self.loss_normalizer)
    #
    #     # regression loss
    #     loss_box_reg = smooth_l1_loss(
    #         pred_anchor_deltas[foreground_idxs],
    #         gt_anchors_deltas[foreground_idxs],
    #         beta=self.smooth_l1_loss_beta,
    #         reduction="sum",
    #     ) / max(1, self.loss_normalizer)
    #
    #     return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    # @torch.no_grad()
    # def get_ground_truth(self, anchors, targets):
    #     """
    #     Args:
    #         anchors (list[Boxes]): A list of #feature level Boxes.
    #             The Boxes contains anchors of this image on the specific feature level.
    #         targets (list[Instances]): a list of N `Instances`s. The i-th
    #             `Instances` contains the ground-truth per-instance annotations
    #             for the i-th input image.  Specify `targets` during training only.
    #
    #     Returns:
    #         gt_classes (Tensor):
    #             An integer tensor of shape (N, R) storing ground-truth
    #             labels for each anchor.
    #             R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
    #             Anchors with an IoU with some target higher than the foreground threshold
    #             are assigned their corresponding label in the [0, K-1] range.
    #             Anchors whose IoU are below the background threshold are assigned
    #             the label "K". Anchors whose IoU are between the foreground and background
    #             thresholds are assigned a label "-1", i.e. ignore.
    #         gt_anchors_deltas (Tensor):
    #             Shape (N, R, 4).
    #             The last dimension represents ground-truth box2box transform
    #             targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
    #             The values in the tensor are meaningful only when the corresponding
    #             anchor is labeled as foreground.
    #     """
    #     gt_classes = []
    #     gt_anchors_deltas = []
    #     anchors = Boxes.cat(anchors)  # Rx4
    #
    #     for targets_per_image in targets:
    #         match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors)
    #         gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)
    #
    #         has_gt = len(targets_per_image) > 0
    #         if has_gt:
    #             # ground truth box regression
    #             matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
    #             gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
    #                 anchors.tensor, matched_gt_boxes.tensor
    #             )
    #
    #             gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
    #             # Anchors with label 0 are treated as background.
    #             gt_classes_i[anchor_labels == 0] = self.num_classes
    #             # Anchors with label -1 are ignored.
    #             gt_classes_i[anchor_labels == -1] = -1
    #         else:
    #             gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
    #             gt_anchors_reg_deltas_i = torch.zeros_like(anchors.tensor)
    #
    #         gt_classes.append(gt_classes_i)
    #         gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
    #
    #     return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)
    #
    # def inference(self, box_cls, box_delta, anchors, image_sizes):
    #     """
    #     Arguments:
    #         box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
    #         anchors (list[Boxes]): A list of #feature level Boxes.
    #             The Boxes contain anchors of this image on the specific feature level.
    #         image_sizes (List[torch.Size]): the input image sizes
    #
    #     Returns:
    #         results (List[Instances]): a list of #images elements.
    #     """
    #     results = []
    #
    #     box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
    #     box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    #     # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)
    #
    #     for img_idx, image_size in enumerate(image_sizes):
    #         box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
    #         box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
    #         results_per_image = self.inference_single_image(
    #             box_cls_per_image, box_reg_per_image, anchors, tuple(image_size)
    #         )
    #         results.append(results_per_image)
    #     return results
    #
    # def inference_single_image(self, box_cls, box_delta, anchors, image_size):
    #     """
    #     Single-image inference. Return bounding-box detection results by thresholding
    #     on scores and applying non-maximum suppression (NMS).
    #
    #     Arguments:
    #         box_cls (list[Tensor]): list of #feature levels. Each entry contains
    #             tensor of size (H x W x A, K)
    #         box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
    #         anchors (list[Boxes]): list of #feature levels. Each entry contains
    #             a Boxes object, which contains all the anchors for that
    #             image in that feature level.
    #         image_size (tuple(H, W)): a tuple of the image height and width.
    #
    #     Returns:
    #         Same as `inference`, but for only one image.
    #     """
    #     boxes_all = []
    #     scores_all = []
    #     class_idxs_all = []
    #
    #     # Iterate over every feature level
    #     for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
    #         # (HxWxAxK,)
    #         box_cls_i = box_cls_i.flatten().sigmoid_()
    #
    #         # Keep top k top scoring indices only.
    #         num_topk = min(self.topk_candidates, box_reg_i.size(0))
    #         # torch.sort is actually faster than .topk (at least on GPUs)
    #         predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
    #         predicted_prob = predicted_prob[:num_topk]
    #         topk_idxs = topk_idxs[:num_topk]
    #
    #         # filter out the proposals with low confidence score
    #         keep_idxs = predicted_prob > self.score_threshold
    #         predicted_prob = predicted_prob[keep_idxs]
    #         topk_idxs = topk_idxs[keep_idxs]
    #
    #         anchor_idxs = topk_idxs // self.num_classes
    #         classes_idxs = topk_idxs % self.num_classes
    #
    #         box_reg_i = box_reg_i[anchor_idxs]
    #         anchors_i = anchors_i[anchor_idxs]
    #         # predict boxes
    #         predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
    #
    #         boxes_all.append(predicted_boxes)
    #         scores_all.append(predicted_prob)
    #         class_idxs_all.append(classes_idxs)
    #
    #     boxes_all, scores_all, class_idxs_all = [
    #         cat(x) for x in [boxes_all, scores_all, class_idxs_all]
    #     ]
    #     keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
    #     keep = keep[: self.max_detections_per_image]
    #
    #     result = Instances(image_size)
    #     result.pred_boxes = Boxes(boxes_all[keep])
    #     result.scores = scores_all[keep]
    #     result.pred_classes = class_idxs_all[keep]
    #     return result


class EfficientDetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        tower_repeat = [3, 3, 3, 4, 4, 4, 5, 5]
        in_channels      = input_shape[0].channels
        compound_coef = cfg.MODEL.EFFICIENTNET.COMPOUND_COEFFICIENT
        num_classes      = cfg.MODEL.EFFICIENTDET.NUM_CLASSES
        num_convs        = cfg.MODEL.EFFICIENTDET.NUM_CONVS
        num_convs        = tower_repeat[compound_coef] if num_convs < 0 else num_convs
        prior_prob       = cfg.MODEL.EFFICIENTDET.PRIOR_PROB
        num_anchors      = build_anchor_generator(cfg, input_shape).num_cell_anchors
        norm             = cfg.MODEL.EFFICIENTDET.NORM
        export_onnx      = cfg.MODEL.EXPORT_ONNX

        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                SeparableConvBlock(in_channels, norm=norm, activation=True, onnx_export=export_onnx)
            )
            bbox_subnet.append(
                SeparableConvBlock(in_channels, norm=norm, activation=True, onnx_export=export_onnx)
            )

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = SeparableConvBlock(in_channels, num_anchors * num_classes)
        self.bbox_pred = SeparableConvBlock(in_channels, num_anchors * 4)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.pointwise_conv.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg