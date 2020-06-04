#!/usr/bin/env python3
# @Time    : 27/5/20 3:52 PM
# @Author  : fangcheng.ji
# @FileName: classification_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from fvcore.nn import sigmoid_focal_loss_jit

from .classification_head import ClassificationHead


@META_ARCH_REGISTRY.register()
class ClassificationNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)

        self.num_classes = cfg.MODEL.CLASSIFICATION.NUM_CLASSES
        self.in_features = cfg.MODEL.CLASSIFICATION.IN_FEATURES
        self.score_thresh = cfg.MODEL.CLASSIFICATION.SCORE_THRESH
        self.activation = cfg.MODEL.CLASSIFICATION.ACTIVATION

        self.focal_loss_alpha = cfg.MODEL.CLASSIFICATION.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.CLASSIFICATION.FOCAL_LOSS_GAMMA

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = ClassificationHead(cfg, feature_shapes)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # for onnx export
        self.export_onnx = cfg.MODEL.EXPORT_ONNX

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        if self.export_onnx:
            features = self.backbone(batched_inputs)
            features = [features[f] for f in self.in_features]
            cls_logits = self.head(features)
            return cls_logits

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        features = [features[f] for f in self.in_features]
        cls_logits = self.head(features)

        if "class_id" in batched_inputs[0]:
            gt_cls = [x["class_id"].to(self.device) for x in batched_inputs]

        if self.training:
            gt_target = self.get_ground_truth(gt_cls)
            # print("file", [(x['file_name']).split('/')[-1] for x in batched_inputs])
            # print("gt", gt_target.detach().flatten())
            # print("pred", (F.sigmoid(cls_logits.detach())).flatten())

            cls_loss = self.losses(gt_target, cls_logits)
            return cls_loss
        else:
            batch_num = cls_logits.size()[0]
            pred_cls = self.inference(cls_logits)
            results = []
            for i in range(batch_num):
                results.append({"class_id": pred_cls[i]})
            return results

    @torch.no_grad()
    def get_ground_truth(self, gt_cls):
        channel = self.num_classes if self.activation != "softmax" else self.num_classes + 1
        gt_classes_target = torch.zeros(len(gt_cls), channel, device=self.device)
        for i in range(len(gt_cls)):
            if gt_cls[i] >= 0:
                gt_classes_target[i, gt_cls[i]] = 1.0
            elif self.activation == "softmax":      # if cls = -1 and softmax, set cls_id = num_classes
                gt_classes_target[i, self.num_classes] = 1.0
        return gt_classes_target

    def losses(self, gt_target, pred_logits):
        num_batchs = pred_logits.size()[0]

        if self.activation == 'softmax':
            pass
            #     gt_category = torch.argmax(gt_classes[self.classification_tasks[0]].view(num_batchs, -1), dim=1)
            #     valid_category = valid_category.view(num_batchs, -1).sum(dim=1) > 0
            #     loss = F.cross_entropy(
            #         pred_category[valid_category],
            #         gt_category[valid_category],
            #         reduction="sum",
            #     ) / max(1, valid_category.sum())
        else:
            # calculate loss
            loss = F.binary_cross_entropy_with_logits(
                pred_logits.flatten(),
                gt_target.flatten()
            )

            # loss = sigmoid_focal_loss_jit(
            #     pred_logits.flatten(),
            #     gt_target.flatten(),
            #     alpha=self.focal_loss_alpha,
            #     gamma=self.focal_loss_gamma,
            #     reduction="sum",
            # ) / max(1, num_batchs)

        return {"cls_loss": loss}

    def inference(self, pred_logits):
        batch_nums = pred_logits.size()[0]
        if self.activation == 'sigmoid':
            pred_logits.sigmoid_().view(batch_nums, -1)
            values, indices = pred_logits.max(dim=1)
            indices[values < self.score_thresh] = -1
            indices = indices.to(torch.float32)
            result = torch.stack((indices, values), dim=1)
        elif self.activation == 'softmax':
            pred_logits = F.softmax(pred_logits, dim=1)
            values, indices = pred_logits.max(dim=1)
            indices = indices.to(torch.float32)
            result = torch.stack((indices, values), dim=1)
        else:
            raise Exception('Activation is not implemented!!')

        return result