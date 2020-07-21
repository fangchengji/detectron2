#!/usr/bin/env python3
# @Time    : 19/6/20 11:37 AM
# @Author  : fangcheng.ji
# @FileName: boxes_fusion.py

import torch
import numpy as np

from detectron2.structures import Boxes, Instances

from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion


'''
boxes_list = [[
    [0.00, 0.51, 0.81, 0.91],
    [0.10, 0.31, 0.71, 0.61],
    [0.01, 0.32, 0.83, 0.93],
    [0.02, 0.53, 0.11, 0.94],
    [0.03, 0.24, 0.12, 0.35],
],[
    [0.04, 0.56, 0.84, 0.92],
    [0.12, 0.33, 0.72, 0.64],
    [0.38, 0.66, 0.79, 0.95],
    [0.08, 0.49, 0.21, 0.89],
]]
scores_list = [[0.9, 0.8, 0.2, 0.4, 0.7], [0.5, 0.8, 0.7, 0.3]]
labels_list = [[0, 1, 0, 1, 1], [1, 1, 1, 0]]
weights = [2, 1]

iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1

boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
'''


def boxes_fusion_single_image(
    boxes, scores, classes, image_shape, nms_thresh=0.5, topk_per_image=-1, method='nms', device="cpu"
):
    assert method in ["nms", "wbf"], f"Not implemented method {method}"
    assert len(scores) == len(boxes) and len(scores) == len(classes), \
        f"Length of boxes, scores, classes is not equal!"

    # normalize the boxes
    for i, boxes_per_img in enumerate(boxes):
        boxes_per_img = Boxes(boxes_per_img)
        boxes_per_img.clip(image_shape)
        # filter the width or height < threshold boxes
        keep = boxes_per_img.nonempty(1.0)
        boxes_per_img = boxes_per_img[keep]

        boxes_per_img = boxes_per_img.tensor.cpu().numpy()
        boxes_per_img[:, 0::2] = boxes_per_img[:, 0::2] / image_shape[1]
        boxes_per_img[:, 1::2] = boxes_per_img[:, 1::2] / image_shape[0]

        boxes[i] = boxes_per_img
        scores[i] = scores[i][keep].cpu().numpy()
        classes[i] = classes[i][keep].cpu().numpy()

    # weights = [1.2, 1.2, 1.1, 1.1, 1.0, 1.0]
    if method == 'nms':
        boxes, scores, classes = weighted_boxes_fusion(
            boxes, scores, classes,
            # weights=weights,
            iou_thr=nms_thresh
        )
    else:    # "wbf"
        boxes, scores, classes = weighted_boxes_fusion(
            boxes, scores, classes,
            # weights=weights,
            iou_thr=nms_thresh,    # wbf higher than nms performance better
        )

    if topk_per_image >= 0:
        boxes, scores, classes = boxes[:topk_per_image], scores[:topk_per_image], classes[:topk_per_image]

    # resize to image shape
    boxes[:, 0::2] = boxes[:, 0::2] * image_shape[1]
    boxes[:, 1::2] = boxes[:, 1::2] * image_shape[0]

    # to tensor
    boxes = torch.from_numpy(boxes).to(device=device)
    scores = torch.from_numpy(scores).to(device=device)
    classes = torch.from_numpy(classes).to(device=device)

    result = Instances(image_shape)
    boxes = Boxes(boxes)
    boxes.clip(image_shape)
    result.pred_boxes = boxes
    result.scores = scores
    result.pred_classes = classes

    return result


    # valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    # if not valid_mask.all():
    #     boxes = boxes[valid_mask]
    #     scores = scores[valid_mask]
    #
    # scores = scores[:, :-1]
    # num_bbox_reg_classes = boxes.shape[1] // 4
    # # Convert to Boxes to use the `clip` function ...
    # boxes = Boxes(boxes.reshape(-1, 4))
    # boxes.clip(image_shape)
    # boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    #
    # # Filter results based on detection scores
    # filter_mask = scores > score_thresh  # R x K
    # # R' x 2. First column contains indices of the R predictions;
    # # Second column contains indices of classes.
    # filter_inds = filter_mask.nonzero()
    # if num_bbox_reg_classes == 1:
    #     boxes = boxes[filter_inds[:, 0], 0]
    # else:
    #     boxes = boxes[filter_mask]
    # scores = scores[filter_mask]
    #
    # # Apply per-class NMS
    # keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    # if topk_per_image >= 0:
    #     keep = keep[:topk_per_image]
    # boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    # result = Instances(image_shape)
    # result.pred_boxes = Boxes(boxes)
    # result.scores = scores
    # result.pred_classes = filter_inds[:, 1]
    #
    # return result, filter_inds[:, 0]

