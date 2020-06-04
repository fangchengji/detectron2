#!/usr/bin/env python3
# @Time    : 31/5/20 11:07 AM
# @Author  : fangcheng.ji
# @FileName: dataset_mapper.py

import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
import random

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures.boxes import BoxMode
from detectron2.data.dataset_mapper import DatasetMapper

from .augmentation import random_affine, augment_hsv


class PlusDatasetMapper:
    """
    a CALLABLE WHICH TAKES A DATASET DICT IN dETECTRON2 dATASET FORMAT,
    AND MAP IT INTO A FORMAT USED BY THE MODEL.

    tHIS IS THE DEFAULT CALLABLE TO BE USED TO MAP YOUR DATASET DICT INTO TRAINING DATA.
    yOU MAY NEED TO FOLLOW IT TO IMPLEMENT YOUR OWN ONE FOR CUSTOMIZED LOGIC,
    SUCH AS A DIFFERENT WAY TO READ OR TRANSFORM IMAGES.
    sEE :DOC:`/TUTORIALS/DATA_LOADING` FOR DETAILS.

    tHE CALLABLE CURRENTLY DOES THE FOLLOWING:

    1. rEAD THE IMAGE FROM "FILE_NAME"
    2. aPPLIES CROPPING/GEOMETRIC TRANSFORMS TO THE IMAGE AND ANNOTATIONS
    3. pREPARE DATA AND ANNOTATIONS TO tENSOR AND :CLASS:`iNSTANCES`
    """
    def __init__(self, cfg, dataset, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = self.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        self._dataset = dataset

        # for multi image augmentation
        self.mosaic_prob = cfg.DATALOADER.MOSAIC_AUGMENTATION
        self.mixup_prob = cfg.DATALOADER.MIXUP_AUGMENTATION
        self.hsv_prob = cfg.DATALOADER.HSV_AUGMENTATION

    def build_transform_gen(self, cfg, is_train):
        """
        Create a list of :class:`TransformGen` from config.
        Now it includes resizing and flipping.

        Returns:
            list[TransformGen]
        """
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
        if sample_style == "range":
            assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
                len(min_size)
            )

        logger = logging.getLogger(__name__)
        tfm_gens = []
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
        if is_train:
            tfm_gens.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
            tfm_gens.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
            # TODO: add more augmentation
            tfm_gens.append(T.RandomApply(T.RandomBrightness(0.8, 1.2), 0.5))   # 0.8 ~ 1.2 is the default value of albumentations
            tfm_gens.append(T.RandomApply(T.RandomContrast(0.8, 1.2), 0.5))

            logger.info("TransformGens used in training: " + str(tfm_gens))
        return tfm_gens

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        if rand_range() < self.mosaic_prob:
            image, dataset_dict = self.load_mosaic(dataset_dict)
        else:
            image = self.load_image(dataset_dict)

        # augment form yolov4
        if rand_range() < self.hsv_prob:
            image = augment_hsv(image, hgain=0.0138, sgain=0.678, vgain=0.36)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

    def load_image(self, dataset_dict):
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        return image

    def load_mosaic(self, data_dict):
        """
        only support bbox now
        Args:
            data_dict:
        Returns:
        """
        # loads images in a mosaic
        data_dicts = [data_dict]
        data_dicts.extend([copy.deepcopy(self._dataset[random.randint(0, len(self._dataset) - 1)]) for _ in range(3)])
        images = []
        labels = []
        w, h = 10000, 10000
        for data_dict in data_dicts:
            images.append(self.load_image(data_dict))
            if "annotations" in data_dict:
                labels.append(
                    np.array([[obj['category_id']] + obj['bbox'] for obj in data_dict["annotations"]], dtype=np.float32)
                )
            else:
                labels.append(np.array([]))
            w, h = min(w, data_dict["width"]), min(h, data_dict["height"])
        xc, yc = int(random.uniform(w * 0.5, w * 1.5)), int(random.uniform(h * 0.5, h * 1.5))   # mosaic center x, y

        mosaic_img = np.full((h * 2, w * 2, images[0].shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
        mosaic_labels = []
        for i, (image, data_dict) in enumerate(zip(images, data_dicts)):
            wi, hi = data_dict['width'], data_dict['height']

            # place img in mosaic_img
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - wi, 0), max(yc - hi, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = wi - (x2a - x1a), hi - (y2a - y1a), wi, hi  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - hi, 0), min(xc + wi, w * 2), yc
                x1b, y1b, x2b, y2b = 0, hi - (y2a - y1a), min(wi, x2a - x1a), hi
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - wi, 0), yc, xc, min(h * 2, yc + hi)
                x1b, y1b, x2b, y2b = wi - (x2a - x1a), 0, max(xc, wi), min(y2a - y1a, hi)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + wi, w * 2), min(h * 2, yc + hi)
                x1b, y1b, x2b, y2b = 0, 0, min(wi, x2a - x1a), min(y2a - y1a, hi)

            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if len(labels[i]) > 0:
                labels_i = labels[i]
                # xywh to xyxy
                labels_i[:, 1] += padw
                labels_i[:, 2] += padh
                labels_i[:, 3] = labels_i[:, 1] + labels_i[:, 3]
                labels_i[:, 4] = labels_i[:, 2] + labels_i[:, 4]
                mosaic_labels.append(labels_i)

            # if "annotations" in data_dict:
            #     labels = copy.deepcopy(data_dict["annotations"])
            #     for obj in labels:  # in coco x0,y0,w,h mode
            #         obj['bbox'][0] += padw
            #         obj['bbox'][1] += padh
            #         # clip
            #         np.clip(obj['bbox'][0], 0, 2 * w - 1, out=obj['bbox'][0])
            #         np.clip(obj['bbox'][1], 0, 2 * h - 1, out=obj['bbox'][1])
            #         np.clip(obj['bbox'][2], 0, 2 * w - 1 - obj['bbox'][0], out=obj['bbox'][2])
            #         np.clip(obj['bbox'][3], 0, 2 * h - 1 - obj['bbox'][1], out=obj['bbox'][3])
            #         # filter the very small bbox
            #         if obj['bbox'][2] < 8 or obj['bbox'][3] < 8:
            #             continue
            #         annotations.append(obj)

        # Concat/clip labels
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(mosaic_labels[:, [1, 3]], 0, 2 * w, out=mosaic_labels[:, [1, 3]])  # use with random_affine
            np.clip(mosaic_labels[:, [2, 4]], 0, 2 * h, out=mosaic_labels[:, [2, 4]])  # use with random_affine

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        mosaic_img, mosaic_labels = random_affine(
            mosaic_img,
            mosaic_labels,
            degrees=1.98 * 2,
            translate=0.05 * 2,
            scale=0.05 * 2,
            shear=0.641 * 2,
            border=(-h // 2, -w // 2)
        )  # border to remove

        # translate back to detectron2 format
        annos = []
        for i in range(mosaic_labels.shape[0]):
            anno = {
                'iscrowd': 0,
                # xyxy to xywh
                'bbox': [mosaic_labels[i, 1],
                         mosaic_labels[i, 2],
                         max(mosaic_labels[i, 3] - mosaic_labels[i, 1], 0),
                         max(mosaic_labels[i, 4] - mosaic_labels[i, 2], 0)],
                'category_id': int(mosaic_labels[i, 0]),
                'bbox_mode': BoxMode(1)   # xywh
                }
            annos.append(anno)

        data_dict = {
            "file_name": data_dicts[0]['file_name'],
            "height": h,
            "width": w,
            "image_id": data_dicts[0]['image_id'],
            "annotations": annos
        }

        return mosaic_img, data_dict


def rand_range(low=1.0, high=None, size=None):
    """
    Uniform float random number between low and high.
    """
    if high is None:
        low, high = 0, low
    if size is None:
        size = []
    return np.random.uniform(low, high, size)