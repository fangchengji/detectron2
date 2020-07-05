# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
from itertools import count
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform, Transform
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import cv2

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    TransformGen,
    apply_transform_gens,
)
from detectron2.structures import Boxes, Instances

from detectron2.modeling.meta_arch import GeneralizedRCNN, OneStageDetector

from .boxes_fusion import boxes_fusion_single_image


__all__ = ["DatasetMapperTTA", "OneStageDetectorWithTTA"]


class Rotation90Transform(Transform):
    def __init__(self, h, w, angle=90, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        assert angle == 90 or angle == -90, "Only support 90 or -90 angle!"
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = Rotation90Transform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        # for alleria image is 1024 * 1024, don't need crop
        # crop = CropTransform(
        #     (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        # )
        return rotation


class Rotation90Gen(TransformGen):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            return Rotation90Transform(h, w)
        else:
            return NoOpTransform()


class CLAHE(Transform):
    def __init__(self, img_format='BGR'):
        super().__init__()
        self.img_format=img_format

    def apply_image(self, img, clip_limit=2.0, tile_grid_size=(8, 8)):
        if img.dtype != np.uint8:
            raise TypeError("clahe supports only uint8 inputs")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        if len(img.shape) == 2 or img.shape[2] == 1:
            if self.img_format=="BGR":
                img = img[..., ::-1]
            # apply to rgb image
            img = clahe.apply(img)

            if self.img_format=="BGR":
                img = img[..., ::-1]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return img

    def apply_coords(self, coords):
        return coords

    def inverse(self):
        return self


class CLAHEGen(TransformGen):
    def __init__(self, prob=0.5, img_format='BGR'):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        do = self._rand_range() < self.prob
        if do:
            return CLAHE(self.img_format)
        else:
            return NoOpTransform()


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT

        self.vertical_flip = cfg.TEST.AUG.VERTICAL_FLIP

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a detection dataset dict in standard format

        Returns:
            list[dict]:
                a list of dataset dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()

        # Create all combinations of augmentations to use
        tfm_gen_candidates = []  # each element is a list[TransformGen]
        for min_size in self.min_sizes:
            resize = ResizeShortestEdge(min_size, self.max_size)
            tfm_gen_candidates.append([resize])  # resize only
            if self.flip:
                flip = RandomFlip(prob=1.0)
                tfm_gen_candidates.append([resize, flip])  # resize + flip
            if self.vertical_flip:
                flip = RandomFlip(prob=1.0, horizontal=False, vertical=True)
                tfm_gen_candidates.append([resize, flip])  # resize + flip

            tfm_gen_candidates.append([resize, Rotation90Gen(prob=1.0)])
            # tfm_gen_candidates.append([resize, CLAHEGen(prob=1.0, img_format=self.image_format)])
            tfm_gen_candidates.append([resize,
                                       Rotation90Gen(prob=1.0),
                                       Rotation90Gen(prob=1.0),
                                       Rotation90Gen(prob=1.0)])

        # Apply all the augmentations
        ret = []
        for tfm_gen in tfm_gen_candidates:
            new_image, tfms = apply_transform_gens(tfm_gen, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))

            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image

            # pop original image width and height to prevent rescale boxes in postprocess of detector
            dic.pop("width")
            dic.pop("height")

            ret.append(dic)
        return ret


class OneStageDetectorWithTTA(nn.Module):
    """
    A GeneralizedRCNN with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(model, OneStageDetector), \
            "TTA is only supported on OneStageDetector. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

        if 'FCOS' in cfg.MODEL.PROPOSAL_GENERATOR.NAME:
            self.topk_per_image = self.cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
            self.nms_threshold = self.cfg.MODEL.FCOS.NMS_TH
        elif 'ATSS' in cfg.MODEL.PROPOSAL_GENERATOR.NAME:
            self.topk_per_image = self.cfg.MODEL.ATSS.POST_NMS_TOPK_TEST
            self.nms_threshold = self.cfg.MODEL.ATSS.NMS_TH
        else:
            raise ValueError("Not implemented TTA Arch")

        self.device = self.model.device

    # @contextmanager
    # def _turn_off_roi_heads(self, attrs):
    #     """
    #     Open a context where some heads in `model.roi_heads` are temporarily turned off.
    #     Args:
    #         attr (list[str]): the attribute in `model.roi_heads` which can be used
    #             to turn off a specific head, e.g., "mask_on", "keypoint_on".
    #     """
    #     roi_heads = self.model.roi_heads
    #     old = {}
    #     for attr in attrs:
    #         try:
    #             old[attr] = getattr(roi_heads, attr)
    #         except AttributeError:
    #             # The head may not be implemented in certain ROIHeads
    #             pass
    #
    #     if len(old.keys()) == 0:
    #         yield
    #     else:
    #         for attr in old.keys():
    #             setattr(roi_heads, attr, False)
    #         yield
    #         for attr in old.keys():
    #             setattr(roi_heads, attr, old[attr])

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        # """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.forward(inputs)
                )
                inputs, instances = [], []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.image_format)
                image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        # with self._turn_off_roi_heads(["mask_on", "keypoint_on"]):
            # temporarily disable roi heads
        all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)

        return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_boxes(self, augmented_inputs, tfms):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for output, tfm in zip(outputs, tfms):
            # Need to inverse the transforms on boxes, to obtain results on original image
            if isinstance(output, dict) and 'instances' in output:
                output = output['instances']
            pred_boxes = output.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))
            # all_boxes.append(original_pred_boxes)
            all_scores.append(output.scores)
            all_classes.append(output.pred_classes)
        # all_boxes = torch.cat(all_boxes, dim=0)
        return all_boxes, all_scores, all_classes

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):

        merged_instances = boxes_fusion_single_image(
            all_boxes, all_scores, all_classes, shape_hw,
            self.nms_threshold,
            self.topk_per_image,
            method="wbf",
            device=self.device,
        )

        # # select from the union of all results
        # num_boxes = len(all_boxes)
        # # num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # num_classes = self.cfg.MODEL.FCOS.NUM_CLASSES
        # # +1 because fast_rcnn_inference expects background scores as well
        # all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        # for idx, cls, score in zip(count(), all_classes, all_scores):
        #     all_scores_2d[idx, cls] = score
        #
        # merged_instances, _ = fast_rcnn_inference_single_image(
        #     all_boxes,
        #     all_scores_2d,
        #     shape_hw,
        #     1e-8,
        #     self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        #     self.cfg.TEST.DETECTIONS_PER_IMAGE,
        # )

        return merged_instances

    def _rescale_detected_boxes(self, augmented_inputs, merged_instances, tfms):
        augmented_instances = []
        for input, tfm in zip(augmented_inputs, tfms):
            # Transform the target box to the augmented image's coordinate space
            pred_boxes = merged_instances.pred_boxes.tensor.cpu().numpy()
            pred_boxes = torch.from_numpy(tfm.apply_box(pred_boxes))

            aug_instances = Instances(
                image_size=input["image"].shape[1:3],
                pred_boxes=Boxes(pred_boxes),
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        return augmented_instances
    #
    # def _reduce_pred_masks(self, outputs, tfms):
    #     # Should apply inverse transforms on masks.
    #     # We assume only resize & flip are used. pred_masks is a scale-invariant
    #     # representation, so we handle flip specially
    #     for output, tfm in zip(outputs, tfms):
    #         if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
    #             output.pred_masks = output.pred_masks.flip(dims=[3])
    #     all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
    #     avg_pred_masks = torch.mean(all_pred_masks, dim=0)
    #     return avg_pred_masks
