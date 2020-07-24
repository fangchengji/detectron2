#!/usr/bin/env python3
# @Time    : 31/5/20 7:49 PM
# @Author  : fangcheng.ji
# @FileName: augmentation.py

import numpy as np
import random
import cv2
import math

import albumentations as A
import albumentations.augmentations.functional as AF


def get_albumentations_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(600, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.8),
            # A.OneOf([
            #     A.RGBShift(p=1.0),
            #     A.CLAHE(p=1.0),  # internal logic is rgb order
            #     A.RandomGamma(p=1.0),
            # ], p=0.4),
            A.CLAHE(p=0.3),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=(124, 117, 104), p=0.5),   # rgb order
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='coco',
            min_area=4,
            min_visibility=0.01,
            label_fields=['category_id']
        )
    )


def get_albumentations_test_transforms():
    return A.Compose(
        [
            # A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.3),
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            # ], p=0.5),
            # A.CLAHE(p=0.1),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='coco',
            min_area=4,
            min_visibility=0.01,
            label_fields=['category_id']
        )
    )


def get_albumentations_infer_transforms():
    # please return None if don't need it any more
    return A.CLAHE(p=1.0)         # internal logic is rgb order
    # return None


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


def augment_brightness_contrast(img, brightness=0.2, contrast=0.2):
    alpha = 1.0 + random.uniform(-contrast, contrast)
    beta = 0.0 + random.uniform(-brightness, brightness)
    return AF.brightness_contrast_adjust(img, alpha, beta, beta_by_max=True)


# class Cutout(A.DualTransform):
#     """CoarseDropout of the square regions in the image.
#     Reference:
#     |  https://arxiv.org/abs/1708.04552
#     |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
#     |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
#     """
#     def __init__(self, p=0.5):
#         super().__init__()
#         self.p = p
#
#         self.holes = []
#         # self.num_holes = num_holes
#         # self.max_h_size = max_h_size
#         # self.max_w_size = max_w_size
#         # self.fill_value = fill_value
#         # warnings.warn("This class has been deprecated. Please use CoarseDropout", DeprecationWarning)
#
#     def apply(self, image, holes=[], **params):
#         h, w = image.shape[:2]
#         # create random masks
#         scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
#         self.holes = []
#         for s in scales:
#             mask_h = random.randint(1, int(h * s))
#             mask_w = random.randint(1, int(w * s))
#
#             # box
#             xmin = max(0, random.randint(0, w) - mask_w // 2)
#             ymin = max(0, random.randint(0, h) - mask_h // 2)
#             xmax = min(w, xmin + mask_w)
#             ymax = min(h, ymin + mask_h)
#
#             self.holes.append([xmin, ymin, xmax, ymax])
#             # apply random color mask
#             image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
#
#         return image
#
#     def apply_to_bboxes(self, bboxes, **params):
#         # return [self.apply_to_bbox(tuple(bbox[:4]), **params) + tuple(bbox[4:]) for bbox in bboxes]
#         # def apply_to_bbox(self, bbox, **params):
#         assert len(self.holes) > 0, "Should generate holes first! "
#
#         def bbox_ioa(box1, box2):
#             # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
#             box2 = box2.transpose()
#
#             # Get the coordinates of bounding boxes
#             b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#             b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#
#             # Intersection area
#             inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
#                          (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
#
#             # box2 area
#             box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16
#
#             # Intersection over box2 area
#             return inter_area / box2_area
#
#         for hole in self.holes:
#             # return unobscured labels
#             if len(bboxes) > 0:
#                 box = np.array(hole, dtype=np.float32)
#                 ioa = bbox_ioa(box, bboxes[:, :4])  # intersection over area
#                 bboxes = bboxes[ioa < 0.60]  # remove >60% obscured labels
#
#         # clear the holes
#         self.holes = []
#         return bboxes


def cutout(image, dataset_dict):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]
    image = image.copy()

    annos = dataset_dict["annotations"]
    bboxes = np.array([x['bbox'] for x in annos])
    # xywh to x1y1x2y2
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    box_mask = np.full(len(annos), True)
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(bboxes) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, bboxes)  # intersection over area
            # bboxes = bboxes[ioa < 0.60]  # remove >60% obscured labels
            box_mask = box_mask & (ioa < 0.6)
    #     image.setflags(write=0)
    filter_annos = []
    for i, x in enumerate(annos):
        if box_mask[i]:
            filter_annos.append(x)
    dataset_dict["annotations"] = filter_annos

    return image, dataset_dict


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets
