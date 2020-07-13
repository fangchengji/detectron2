# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


from detectron2.modeling.meta_arch.retinanet import RetinaNetHead

__all__ = ["FashionNetFPN"]


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def permute_all_cls_and_box_to_N_HWA_K_and_concat(box_cls, box_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


@META_ARCH_REGISTRY.register()
class FashionNetFPN(nn.Module):
    """
    Implement FashionNet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # Vis parameters
        self.vis_period               = cfg.VIS_PERIOD
        self.input_format             = cfg.INPUT.FORMAT
        # fmt: on

        # for onnx model export
        self.export_onnx = cfg.MODEL.FASHIONNET.EXPORT_ONNX

        # for classification task
        self.classification_tasks = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.TASK_NAMES
        self.classification_classes = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CLASSES
        assert(len(self.classification_classes) == len(self.classification_tasks))
        self.activation = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.ACTIVATION
        self.fashion_score_threshold = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.SCORE_THRESH

        self.backbone = build_backbone(cfg)
        self.size_divisibility = 32

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]

        self.head = RetinaNetHead(cfg, feature_shapes)
        self.cls_head = FashionClassificationHead(cfg, feature_shapes)

        # # multi task learning with uncertainty
        # self.log_vars = nn.Parameter(torch.zeros(2), requires_grad=True)

        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"].cpu().numpy()
        assert img.shape[0] == 3, "Images should have 3 channels."
        if self.input_format == "BGR":
            img = img[::-1, :, :]
        img = img.transpose(1, 2, 0)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.export_onnx:
            # skip the preprocess
            net_input = batched_inputs
        else:
            # do preprocess
            images = self.preprocess_image(batched_inputs)
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif "targets" in batched_inputs[0]:
                log_first_n(
                    logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
                )
                gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
            else:
                gt_instances = None

            # for fashion classification task
            if "classification" in batched_inputs[0]:
                gt_classification = [x["classification"].to(self.device) for x in batched_inputs]
            else:
                gt_classification = None

            net_input = images.tensor

        features = self.backbone(net_input)
        features = [features[f] for f in self.in_features]

        detection_logits, detection_bbox_reg = self.head(features)
        # classification head
        classification_logits = self.cls_head(features)

        if self.export_onnx:
            # skip the postprocess and return tuple of tensor(onnx needed!)
            return [detection_logits], [detection_bbox_reg], [classification_logits]

        anchors = self.anchor_generator(features)

        if self.training:
            losses = {}
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances, gt_classification)
            # det_loss = self.detection_losses(
            #         gt_classes,
            #         gt_anchors_reg_deltas,
            #         detection_logits,
            #         detection_bbox_reg
            #     )

            losses.update(
                self.detection_losses(
                    gt_classes,
                    gt_anchors_reg_deltas,
                    detection_logits,
                    detection_bbox_reg
                )
            )

            gt_classification_classes = self.get_classification_ground_truth(gt_classification)
            # cls_loss = self.classification_losses(
            #         gt_classification_classes,
            #         classification_logits
            #     )

            losses.update(
                self.classification_losses(
                    gt_classification_classes,
                    classification_logits
                )
            )

            # # multi task learning
            # losses = self.multi_task_learning_loss([det_loss, cls_loss], self.log_vars)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(detection_logits, detection_bbox_reg, anchors, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(detection_logits,
                                     detection_bbox_reg,
                                     anchors,
                                     images.image_sizes)

            category2_results = self.inference_classification(classification_logits)

            processed_results = []
            for results_per_image, input_per_image, image_size, category2 in zip(
                results, batched_inputs, images.image_sizes, category2_results
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                # if category is not commodity or model
                # r = self.filter_output_objects(category2, r)
                processed_results.append({"instances": r, "classification": category2})
            return processed_results

    def multi_task_learning_loss(self, multi_task_losses, uncertainty):
        """
        for multi task learning loss
        Args:
            multi_task_losses: separated multi task loss
            uncertainty: uncertainty tensor for each task

        Returns: losses with uncertainty

        """
        losses = {}
        assert(len(multi_task_losses) == uncertainty.size()[:1], "The losses size is not equal to uncertainty!!")

        alpha = torch.exp(-uncertainty)
        for i, loss in enumerate(multi_task_losses):
            for key, value in loss.items():
                losses[key] = alpha[i] * value
            losses[f"uncertainty_{i}"] = uncertainty[i]

        return losses

    def detection_losses(self, gt_classes, gt_anchors_deltas, pred_class_logits, pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum().item()
        get_event_storage().put_scalar("num_foreground", num_foreground)
        self.loss_normalizer = (
            self.loss_normalizer_momentum * self.loss_normalizer
            + (1 - self.loss_normalizer_momentum) * num_foreground
        )

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        # regression loss
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        return {"loss_cls": 3 * loss_cls, "loss_box_reg": 3 * loss_box_reg}

    def classification_losses(self, gt_classes, pred_class_logits):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, see
                :meth:`RetinaNetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls"
        """
        start = 0
        pred_category = pred_class_logits[:, start: start + self.classification_classes[0]]
        start += self.classification_classes[0]
        pred_part = pred_class_logits[:, start: start + self.classification_classes[1]]
        start += self.classification_classes[1]
        pred_toward = pred_class_logits[:, start: start + self.classification_classes[2]]

        valid_idxs = gt_classes[self.classification_tasks[0]][1::self.classification_classes[0]] == 1
        data_type = pred_category.dtype
        num_batchs = pred_category.size()[0]
        num_model = valid_idxs.sum()

        valid_category = gt_classes[self.classification_tasks[0]][:] > -1
        # category loss
        if valid_category.sum() > 0:
            if self.activation == 'sigmoid':
                loss_category = sigmoid_focal_loss_jit(
                    pred_category.flatten()[valid_category],
                    gt_classes[self.classification_tasks[0]].to(dtype=data_type)[valid_category],
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / max(1, valid_category.sum() / self.classification_classes[0])
            elif self.activation == 'softmax':
                gt_category = torch.argmax(gt_classes[self.classification_tasks[0]].view(num_batchs, -1), dim=1)
                valid_category = valid_category.view(num_batchs, -1).sum(dim=1) > 0
                loss_category = F.cross_entropy(
                    pred_category[valid_category],
                    gt_category[valid_category],
                    reduction="sum",
                ) / max(1, valid_category.sum())
            else:
                raise Exception("Not implement classification activation!")
        else:
            loss_category = 0.0

        valid_part = gt_classes[self.classification_tasks[1]][:] > -1
        if valid_part.sum() > 0:
            loss_part = sigmoid_focal_loss_jit(
                pred_part.flatten()[valid_part],
                gt_classes[self.classification_tasks[1]].to(dtype=data_type)[valid_part],
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / max(1, valid_part.sum() / self.classification_classes[1])
        else:
            loss_part = 0.0

        valid_toward = gt_classes[self.classification_tasks[2]][:] > -1
        if valid_toward.sum() > 0:
            loss_toward = sigmoid_focal_loss_jit(
                pred_toward.flatten()[valid_toward],
                gt_classes[self.classification_tasks[2]].to(dtype=data_type)[valid_toward],
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / max(1, valid_toward.sum() / self.classification_classes[2])
        else:
            loss_toward = 0.0

        return {"loss_category": loss_category,
                "loss_part": loss_part,
                "loss_toward": loss_toward}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets, gt_classification):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = Boxes.cat(anchors)

        for targets_per_image, classification_per_image in zip(targets, gt_classification):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, anchors)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(anchors.tensor)

            # only commodity and model data do object detection,
            # other type ignore all anchors
            # object_detection_enable = classification_per_image.gt_classes == 0 \
            #                           or classification_per_image.gt_classes == 1

            if not has_gt:
                # Anchors with label -1 are ignored.
                gt_classes_i[:] = -1

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    @torch.no_grad()
    def get_classification_ground_truth(self, targets):
        gts = {}
        for task in self.classification_tasks:
            gts[task] = torch.tensor([], dtype=torch.int32, device=self.device)

        for anno in targets:
            for task, num_classes in zip(self.classification_tasks, self.classification_classes):
                tmp = torch.zeros(num_classes, dtype=torch.int32, device=self.device)
                if task == "category":
                    idx = anno.gt_classes
                    # ignore the annotation
                    if anno.gt_ignore == 1:
                        idx = -1
                elif task == "part":
                    idx = anno.gt_part
                    idx -= 1
                elif task == "toward":
                    idx = anno.gt_toward
                    idx -= 1

                if idx >= 0:
                    tmp[idx] = 1
                else:
                    # ignore the value
                    tmp[:] = -1

                gts[task] = torch.cat((gts[task], tmp))

        return gts

    def inference_classification(self, pred_logits):
        results = []
        if self.activation == 'sigmoid':
            pred_logits.sigmoid_()
        elif self.activation == 'softmax':
            pred_logits[:, : self.classification_classes[0]] = F.softmax(
                pred_logits[:, : self.classification_classes[0]],
                dim=1
            )
            pred_logits[:, self.classification_classes[0]:].sigmoid_()
        else:
            raise Exception('Activation is not implemented!!')

        batch_nums = pred_logits.size()[0]

        start = 0
        pred_category = pred_logits[:, start: start + self.classification_classes[0]]
        start += self.classification_classes[0]
        pred_part = pred_logits[:, start: start + self.classification_classes[1]]
        start += self.classification_classes[1]
        pred_toward = pred_logits[:, start: start + self.classification_classes[2]]

        pred_category_id = torch.argmax(pred_category, dim=1).view(batch_nums, -1)
        pred_category_score = torch.gather(pred_category, 1, pred_category_id).view(batch_nums, -1)

        pred_part_id = torch.argmax(pred_part, dim=1).view(batch_nums, -1)
        pred_part_score = torch.gather(pred_part, 1, pred_part_id).view(batch_nums, -1)

        pred_toward_id = torch.argmax(pred_toward, dim=1).view(batch_nums, -1)
        pred_toward_score = torch.gather(pred_toward, 1, pred_toward_id).view(batch_nums, -1)

        for i in range(0, batch_nums):
            result = Instances((0, 0))
            if pred_category_score[i] > self.fashion_score_threshold:
                result.category_id = pred_category_id[i]
                result.category_score = pred_category_score[i]
            else:
                # if score < threshold, let it as the last class.
                result.category_id = torch.tensor([self.classification_classes[0] - 1], device=self.device)
                result.category_score = torch.tensor([0], device=self.device)
            result.part = pred_part_id[i]
            result.part_score = pred_part_score[i]
            result.toward = pred_toward_id[i]
            result.toward_score = pred_toward_score[i]
            results.append(result)

        return results

    def inference(self, box_cls, box_delta, anchors, image_sizes):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # assert len(anchors) == len(image_sizes)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        # for img_idx, anchors_per_image in enumerate(anchors):
        #     image_size = image_sizes[img_idx]
        #     box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
        #     box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
        #     results_per_image = self.inference_single_image(
        #         box_cls_per_image, box_reg_per_image, anchors_per_image, tuple(image_size)
        #     )
        #     results.append(results_per_image)
        # return results

        for img_idx, image_size in enumerate(image_sizes):
            pred_logits_per_image = [x[img_idx] for x in box_cls]
            deltas_per_image = [x[img_idx] for x in box_delta]
            results_per_image = self.inference_single_image(
                pred_logits_per_image, deltas_per_image, anchors, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def filter_output_objects(self, classification, detection):
        # if category is not commodity or model,
        # delete the objects
        if classification.category_id > 1:
            detection.remove('pred_boxes')
            detection.remove('scores')
            detection.remove('pred_classes')
            detection.pred_boxes = Boxes(torch.randn(0, 4, device=self.device))
            detection.scores = torch.tensor([], device=self.device)
            detection.pred_classes = torch.tensor([], device=self.device)
        return detection


class FashionClassificationHead(nn.Module):
    """
    The head used in FashionNet for pictures classification.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = 0
        for shape in input_shape:
            in_channels += shape.channels
        num_classes = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CLASSES
        num_convs = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.NUM_CONVS
        prior_prob = cfg.MODEL.FASHIONNET.CLASSIFICATION_HEAD.PRIOR_PROB
        # fmt: on
        # for multi classification tasks
        if isinstance(num_classes, list):
            self.total_classes = sum(num_classes)
        elif isinstance(num_classes, int):
            self.total_classes = num_classes
        else:
            raise Exception("FashionNet FASHIONNET.CLASSIFICATION_HEAD.NUM_CLASSES needs list or int.")

        # resnet fc for classification tasks
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channels, self.total_classes)

        # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "The 1000-way fully-connected layer is initialized by
        # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
        nn.init.normal_(self.linear.weight, std=0.01)


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
        """
        xs = []
        for feature in features:
            x = self.avgpool(feature)
            x = torch.flatten(x, 1)
            xs.append(x)

        ff = torch.cat(xs, dim=1)
        logits = self.linear(ff)

        return logits

