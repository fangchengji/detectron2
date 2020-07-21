# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
# from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.config import get_cfg

from .data.augmentation import get_albumentations_infer_transforms
from .config import add_alleria_config
from .boxes_fusion import boxes_fusion_single_image


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

        self._albumentation_tfm = get_albumentations_infer_transforms()

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            # 1. apply albumentation transfrom
            # if self._albumentation_tfm is not None:
            #     # original image is default BGR, read by cv. translate to RGB
            #     original_image = original_image[:, :, ::-1]
            #     # need rgb image
            #     original_image = self._albumentation_tfm(image=original_image)['image']
            #     # translate back to rgb
            #     original_image = original_image[:, :, ::-1]

            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class TTAPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        from .test_time_augmentation import OneStageDetectorWithTTA

        # self.tta_mapper = DatasetMapperTTA(cfg)
        self.meta_arch = OneStageDetectorWithTTA(cfg, self.model)

    def __call__(self, original_image):
        with torch.no_grad():
            # 1. apply albumentation transfrom
            # if self._albumentation_tfm is not None:
            #     # original image is default BGR, read by cv. translate to RGB
            #     original_image = original_image[:, :, ::-1]
            #     # need rgb image
            #     original_image = self._albumentation_tfm(image=original_image)['image']
            #     # translate back to rgb
            #     original_image = original_image[:, :, ::-1]

            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]

            height, width = original_image.shape[:2]
            image = self.transform_gen.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = [{"image": image, "height": height, "width": width}]

            # 2. run TTA Inference
            results = self.meta_arch(inputs)[0]
            return results


class EnsemblePredictor:
    def __init__(self, cfg):
        assert cfg.TEST.ENSEMBLE.NUM - 1 == len(cfg.TEST.ENSEMBLE.CONFIGS), "ensemble nums should equal to configs!"
        self.predictors = [TTAPredictor(cfg) if cfg.TEST.AUG.ENABLED else DefaultPredictor(cfg)]

        if 'FCOS' in cfg.MODEL.PROPOSAL_GENERATOR.NAME:
            self.topk_per_image = cfg.MODEL.FCOS.POST_NMS_TOPK_TEST
            self.nms_threshold = cfg.MODEL.FCOS.NMS_TH
        elif 'ATSS' in cfg.MODEL.PROPOSAL_GENERATOR.NAME:
            self.topk_per_image = cfg.MODEL.ATSS.POST_NMS_TOPK_TEST
            self.nms_threshold = cfg.MODEL.ATSS.NMS_TH
        else:
            raise ValueError("Not implemented TTA Arch")

        # init predictors
        for f in cfg.TEST.ENSEMBLE.CONFIGS:
            cfg_i = self.get_config(f)
            self.predictors.append(TTAPredictor(cfg_i) if cfg.TEST.AUG.ENABLED else DefaultPredictor(cfg_i))

    def get_config(self, config_file):
        cfg = get_cfg()
        add_alleria_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.freeze()
        return cfg

    def __call__(self, original_image):
        results = []
        for predictor in self.predictors:
            results.append(predictor(original_image.copy()))
        merged_instances = self.merge_multi_predictions(results, original_image.shape[:2])
        return {"instances": merged_instances}

    def merge_multi_predictions(self, outputs, image_shape):
        all_boxes = []
        all_scores = []
        all_classes = []
        for output in outputs:
            # Need to inverse the transforms on boxes, to obtain results on original image
            if isinstance(output, dict) and 'instances' in output:
                output = output['instances']
            all_boxes.append(output.pred_boxes.tensor)
            all_scores.append(output.scores)
            all_classes.append(output.pred_classes)
        # all_boxes = torch.cat(all_boxes, dim=0)

        merged_instances = boxes_fusion_single_image(
            all_boxes, all_scores, all_classes, image_shape,
            self.nms_threshold,
            self.topk_per_image,
            method="wbf",
            device=all_boxes[0].device,
        )
        return merged_instances


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        if cfg.TEST.ENSEMBLE.ENABLED:
            self.predictor = EnsemblePredictor(cfg)
        elif cfg.TEST.AUG.ENABLED:
            self.predictor = TTAPredictor(cfg)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

