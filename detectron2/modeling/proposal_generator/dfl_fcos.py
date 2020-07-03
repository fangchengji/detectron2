import math
from typing import List, Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat, ml_nms, diou_nms
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY

from detectron2.layers import DFConv2d, FCOSIOULoss, get_norm, quality_focal_loss, distribution_focal_loss
from detectron2.structures import Instances, Boxes
from detectron2.utils.comm import get_world_size, reduce_sum
# from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.layers import sepc_conv
from detectron2.modeling.backbone.sepc import iBN


__all__ = ["DFLFCOS"]

INF = 100000000


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


@PROPOSAL_GENERATOR_REGISTRY.register()
class DFLFCOS(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features          = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides          = cfg.MODEL.FCOS.FPN_STRIDES
        self.focal_loss_alpha     = cfg.MODEL.FCOS.LOSS_ALPHA
        self.focal_loss_gamma     = cfg.MODEL.FCOS.LOSS_GAMMA
        self.center_sample        = cfg.MODEL.FCOS.CENTER_SAMPLE
        self.strides              = cfg.MODEL.FCOS.FPN_STRIDES
        self.radius               = cfg.MODEL.FCOS.POS_RADIUS
        self.pre_nms_thresh_train = cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        self.pre_nms_thresh_test  = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        self.pre_nms_topk_train   = cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN
        self.pre_nms_topk_test    = cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST
        self.nms_thresh           = cfg.MODEL.FCOS.NMS_TH
        self.post_nms_topk_train  = cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        self.post_nms_topk_test   = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
        self.thresh_with_ctr      = cfg.MODEL.FCOS.THRESH_WITH_CTR
        self.mask_on              = cfg.MODEL.MASK_ON #ywlee
        # fmt: on
        self.iou_loss = FCOSIOULoss(cfg.MODEL.FCOS.LOC_LOSS_TYPE)
        self.nms_type = cfg.MODEL.FCOS.NMS_TYPE
        # generate sizes of interest
        soi = []
        prev_size = -1
        for s in cfg.MODEL.FCOS.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi
        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])
        self.fcos_output = FCOSOutputs(
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss,
            self.center_sample,
            self.sizes_of_interest,
            self.strides,
            self.radius,
            self.fcos_head.num_classes,
            self.nms_thresh,
            self.thresh_with_ctr,
            nms_type=self.nms_type,
        )

        # for onnx model export
        self.export_onnx = cfg.MODEL.EXPORT_ONNX

    def forward(self, images, features, gt_instances):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        features = [features[f] for f in self.in_features]
        locations = self.compute_locations(features)
        logits_pred, reg_pred, bbox_towers, scales = self.fcos_head(features)

        if self.export_onnx:
            return logits_pred, reg_pred

        if self.training:
            pre_nms_thresh = self.pre_nms_thresh_train
            pre_nms_topk = self.pre_nms_topk_train
            post_nms_topk = self.post_nms_topk_train
        else:
            pre_nms_thresh = self.pre_nms_thresh_test
            pre_nms_topk = self.pre_nms_topk_test
            post_nms_topk = self.post_nms_topk_test

        self.fcos_output(
            images,
            locations,
            logits_pred,
            reg_pred,
            # ctrness_pred,
            pre_nms_thresh,
            pre_nms_topk,
            post_nms_topk,
            gt_instances,
            scales,
        )

        if self.training:
            losses, _ = self.fcos_output.losses()
            if self.mask_on:
                proposals = self.fcos_output.predict_proposals()
                return proposals, losses
            else:
                return None, losses
        else:
            proposals = self.fcos_output.predict_proposals()
            return proposals, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class FCOSHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        # TODO: Implement the sigmoid version first.
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        head_configs = {"cls": (cfg.MODEL.FCOS.NUM_CLS_CONVS,
                                False),
                        "bbox": (cfg.MODEL.FCOS.NUM_BOX_CONVS,
                                 cfg.MODEL.FCOS.USE_DEFORMABLE),
                        "share": (cfg.MODEL.FCOS.NUM_SHARE_CONVS,
                                  cfg.MODEL.FCOS.USE_DEFORMABLE)}
        norm = None if cfg.MODEL.FCOS.NORM == "none" else cfg.MODEL.FCOS.NORM

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        self.reg_max = 8  # fcos max distance is 8

        # self.lconv = sepc_conv(256, 256, kernel_size=3, dilation=1, padding=1, part_deform=True)
        # self.cconv = sepc_conv(256, 256, kernel_size=3, dilation=1, padding=1, part_deform=True)
        # self.lbn = nn.BatchNorm2d(256)
        # self.cbn = nn.BatchNorm2d(256)

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
            in_channels, self.num_classes,
            kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * (self.reg_max + 1), kernel_size=3,
            stride=1, padding=1
        )
        # self.ctrness = nn.Conv2d(
        #     in_channels, 1, kernel_size=3,
        #     stride=1, padding=1
        # )

        if cfg.MODEL.FCOS.USE_SCALE:
            self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in self.fpn_strides])
        else:
            self.scales = None

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower, self.cls_logits,
            self.bbox_pred,
            # self.ctrness,
            # self.lconv, self.cconv,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        # ctrness = []
        bbox_towers = []
        scales = []

        # # sepc code
        # cls = [self.cconv(level, item) for level, item in enumerate(x)]
        # loc = [self.lconv(level, item) for level, item in enumerate(x)]
        # cls = iBN(cls, self.cbn)
        # loc = iBN(loc, self.lbn)
        # sepc_outs = [[F.relu(s), F.relu(l)] for s, l in zip(cls, loc)]

        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            # ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                # reg = self.scales[l](reg)
                scales.append(self.scales[l])
            # Note that we use relu, as in the improved FCOS, instead of exp.
            # bbox_reg.append(F.relu(reg))
            bbox_reg.append(reg)

        return logits, bbox_reg, bbox_towers, scales


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    labels: refers to the ground-truth class of an position.

    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the ground-truth box.

    logits_pred: predicted classification scores in [-inf, +inf];

    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets 

    ctrness_pred: predicted centerness scores

"""


def compute_ctrness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    ctrness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
              (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(ctrness)


class FCOSOutputs(object):
    def __init__(
            self,
            # images,
            # locations,
            # logits_pred,
            # reg_pred,
            # ctrness_pred,
            focal_loss_alpha,
            focal_loss_gamma,
            iou_loss,
            center_sample,
            sizes_of_interest,
            strides,
            radius,
            num_classes,
            # pre_nms_thresh,
            # pre_nms_top_n,
            nms_thresh,
            # fpn_post_nms_top_n,
            thresh_with_ctr,
            # gt_instances=None,
            nms_type='nms',
    ):
        # self.logits_pred = logits_pred
        # self.reg_pred = reg_pred
        # self.ctrness_pred = ctrness_pred
        # self.locations = locations

        # self.gt_instances = gt_instances
        # self.num_feature_maps = len(logits_pred)
        # self.num_images = len(images)
        # self.image_sizes = images.image_sizes
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.iou_loss = iou_loss
        self.center_sample = center_sample
        self.sizes_of_interest = sizes_of_interest
        self.strides = strides
        self.radius = radius
        self.num_classes = num_classes
        # self.pre_nms_thresh = pre_nms_thresh
        # self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        # self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.thresh_with_ctr = thresh_with_ctr
        self.nms_type = nms_type

        # for dfl
        self.reg_max = 8
        self.distribution_project = Project(self.reg_max)

    def __call__(
        self,
        images,
        locations,
        logits_pred,
        reg_pred,
        # ctrness_pred,
        pre_nms_thresh,
        pre_nms_top_n,
        fpn_post_nms_top_n,
        gt_instances=None,
        scales=None,
    ):
        self.logits_pred = logits_pred
        self.reg_pred = reg_pred
        # self.ctrness_pred = ctrness_pred
        self.locations = locations

        self.gt_instances = gt_instances
        self.num_feature_maps = len(logits_pred)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes

        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n

        self.scales = scales

    def _transpose(self, training_targets, num_loc_list):
        '''
        This function is used to transpose image first training targets to level first ones
        :return: level first training targets
        '''
        for im_i in range(len(training_targets)):
            training_targets[im_i] = torch.split(
                training_targets[im_i], num_loc_list, dim=0
            )

        targets_level_first = []
        for targets_per_level in zip(*training_targets):
            targets_level_first.append(
                torch.cat(targets_per_level, dim=0)
            )
        return targets_level_first

    def _get_ground_truth(self):
        num_loc_list = [len(loc) for loc in self.locations]
        self.num_loc_list = num_loc_list

        # compute locations to size ranges
        loc_to_size_range = []
        for l, loc_per_level in enumerate(self.locations):
            loc_to_size_range_per_level = loc_per_level.new_tensor(self.sizes_of_interest[l])
            loc_to_size_range.append(
                loc_to_size_range_per_level[None].expand(num_loc_list[l], -1)
            )

        loc_to_size_range = torch.cat(loc_to_size_range, dim=0)
        locations = torch.cat(self.locations, dim=0)

        training_targets = self.compute_targets_for_locations(
            locations, self.gt_instances, loc_to_size_range
        )

        # transpose im first training_targets to level first ones
        training_targets = {
            k: self._transpose(v, num_loc_list) for k, v in training_targets.items()
        }

        # we normalize reg_targets by FPN's strides here
        reg_targets = training_targets["reg_targets"]
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
            # for dfl
            reg_targets[l] = reg_targets[l].clamp(min=0.0, max=self.reg_max - 0.01)

        return training_targets

    def get_sample_region(self, gt, strides, num_loc_list, loc_xs, loc_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(loc_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x.numel() == 0 or center_x[..., 0].sum() == 0:
            return loc_xs.new_zeros(loc_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, num_loc in enumerate(num_loc_list):
            end = beg + num_loc
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end
        left = loc_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - loc_xs[:, None]
        top = loc_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - loc_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, size_ranges):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes = targets_per_im.gt_boxes.tensor
            labels_per_im = targets_per_im.gt_classes

            # no gt
            if bboxes.numel() == 0:
                labels.append(labels_per_im.new_zeros(locations.size(0)) + self.num_classes)
                reg_targets.append(locations.new_zeros((locations.size(0), 4)))
                continue

            area = targets_per_im.gt_boxes.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, self.num_loc_list,
                    xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= size_ranges[:, [0]]) & \
                (max_reg_targets_per_im <= size_ranges[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = self.num_classes

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return {"labels": labels, "reg_targets": reg_targets}

    def losses(self):
        """
        Return the losses from a set of FCOS predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
        """

        training_targets = self._get_ground_truth()
        labels, reg_targets = training_targets["labels"], training_targets["reg_targets"]

        # Collect all logits and regression predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W from slowest to fastest axis.
        logits_pred = cat(
            [
                # Reshape: (N, C, Hi, Wi) -> (N, Hi, Wi, C) -> (N*Hi*Wi, C)
                x.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                for x in self.logits_pred
            ], dim=0, )
        reg_pred = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                scale(self.distribution_project(x.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))))
                for x, scale in zip(self.reg_pred, self.scales)
            ], dim=0, )
        reg_pred_dfl = cat(
            [
                # Reshape: (N, B, Hi, Wi) -> (N, Hi, Wi, B) -> (N*Hi*Wi, B)
                x.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
                for x in self.reg_pred
            ], dim=0, )
        # ctrness_pred = cat(
        #     [
        #         # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
        #         x.reshape(-1) for x in self.ctrness_pred
        #     ], dim=0, )

        labels = cat(
            [
                # Reshape: (N, 1, Hi, Wi) -> (N*Hi*Wi,)
                x.reshape(-1) for x in labels
            ], dim=0, )

        reg_targets = cat(
            [
                # Reshape: (N, Hi, Wi, 4) -> (N*Hi*Wi, 4)
                x.reshape(-1, 4) for x in reg_targets
            ], dim=0, )

        return self.fcos_losses(
            labels,
            reg_targets,
            logits_pred,
            reg_pred,
            reg_pred_dfl,
            # ctrness_pred,
            self.focal_loss_alpha,
            self.focal_loss_gamma,
            self.iou_loss
        )

    def fcos_losses(
        self,
        labels,
        reg_targets,
        logits_pred,
        reg_pred,
        reg_pred_dfl,
        # ctrness_pred,
        focal_loss_alpha,
        focal_loss_gamma,
        iou_loss,
    ):
        num_classes = logits_pred.size(1)
        labels = labels.flatten()

        pos_inds = torch.nonzero(labels != num_classes).squeeze(1)
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        total_num_pos = reduce_sum(pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        # prepare one_hot
        class_target = torch.zeros_like(logits_pred)
        class_target[pos_inds, labels[pos_inds]] = 1

        # class_loss = sigmoid_focal_loss_jit(
        #     logits_pred,
        #     class_target,
        #     alpha=focal_loss_alpha,
        #     gamma=focal_loss_gamma,
        #     reduction="sum",
        # ) / num_pos_avg

        reg_pred = reg_pred[pos_inds]
        # pred_distance = self.distribution_project(reg_pred)
        reg_targets = reg_targets[pos_inds]

        # ctrness_targets = compute_ctrness_targets(reg_targets)
        # ctrness_targets_sum = ctrness_targets.sum()
        # ctrness_norm = max(reduce_sum(ctrness_targets_sum).item() / num_gpus, 1e-6)

        # use pred cls score as weight
        weight_targets = logits_pred.detach().sigmoid().max(dim=1)[0][pos_inds]
        weight_targets_sum = weight_targets.sum()
        weight_norm = max(reduce_sum(weight_targets_sum).item() / num_gpus, 1e-6)

        # for quality_focal_loss
        # ctrness_all = torch.zeros(class_target.size()[0], dtype=torch.float32, device=class_target.device)
        # ctrness_all[pos_inds] = ctrness_targets

        score = torch.zeros(class_target.size()[0], dtype=torch.float32, device=class_target.device)
        score[pos_inds] = self.iou_score(reg_pred.detach(), reg_targets)

        class_loss = quality_focal_loss(
            logits_pred,
            class_target,
            score,                # IoU score
            weight=1.0,           # weight = 1.0
            beta=focal_loss_gamma,
            reduction='mean',
            avg_factor=num_pos_avg
        )

        reg_loss = iou_loss(
            reg_pred,
            reg_targets,
            # ctrness_targets,
            weight_targets,
        ) / weight_norm

        # ctrness_loss = F.binary_cross_entropy_with_logits(
        #     ctrness_pred,
        #     ctrness_targets,
        #     reduction="sum"
        # ) / num_pos_avg

        # dfl loss
        pred_ltrb = reg_pred_dfl[pos_inds].reshape(-1, self.reg_max + 1)
        target_ltrb = reg_targets.reshape(-1)
        assert len(target_ltrb) == len(pred_ltrb), "Pred size is not equal to target!"
        loss_dfl = distribution_focal_loss(
            pred_ltrb,
            target_ltrb,
            weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
            avg_factor=4.0 * 4.0,
        ) / weight_norm

        losses = {
            "loss_fcos_cls": class_loss,
            "loss_fcos_loc": reg_loss,
            # "loss_fcos_ctr": ctrness_loss
            "loss_dfl": loss_dfl,
        }
        return losses, {}

    def iou_score(self, pred, target):
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
        return ious

    def predict_proposals(self):
        sampled_boxes = []

        bundle = (
            self.locations, self.logits_pred,
            self.reg_pred,
            # self.ctrness_pred,
            self.strides
        )

        for i, (l, o, r, s) in enumerate(zip(*bundle)):
            # recall that during training, we normalize regression targets with FPN's stride.
            # we denormalize them here.
            r = self.scales[i](self.distribution_project(r.permute(0, 2, 3, 1)))    # N, C, H, W --> N, H, W, C
            r = r * s
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, r, self.image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)
        return boxlists

    def forward_for_single_feature_map(
            self, locations, box_cls,
            reg_pred, image_sizes
    ):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        # box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = reg_pred.reshape(N, -1, 4)
        # ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # ctrness = ctrness.reshape(N, -1).sigmoid()

        # if self.thresh_with_ctr is True, we multiply the classification
        # scores with centerness scores before applying the threshold.
        # if self.thresh_with_ctr:
        #     box_cls = box_cls * ctrness[:, :, None]
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # if not self.thresh_with_ctr:
        #     box_cls = box_cls * ctrness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            boxlist = Instances(image_sizes[i])
            boxes = Boxes(detections)
            boxes.clip(image_sizes[i])
            boxlist.pred_boxes = boxes
            boxlist.scores = torch.sqrt(per_box_cls)
            boxlist.pred_classes = per_class
            boxlist.locations = per_locations

            results.append(boxlist)

        return results

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            if self.nms_type == 'nms' or self.nms_type == 'default':
                result = ml_nms(boxlists[i], self.nms_thresh)
            elif self.nms_type == 'diou_nms':
                result = diou_nms(boxlists[i], self.nms_thresh, beta1=0.9)
            else:
                raise Exception("Not implement nms type!!")
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


class Project(nn.Module):
    """
    A fixed project layer for distribution
    """
    def __init__(self, reg_max=8):
        super(Project, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.to(x.device)).reshape(-1, 4)
        return x