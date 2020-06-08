#!/usr/bin/env python3
# @Time    : 4/6/20 9:34 AM
# @Author  : fangcheng.ji
# @FileName: evaluation.py

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from tabulate import tabulate
from collections import defaultdict
from numba import jit
from tqdm import tqdm

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode, Instances
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco
from detectron2.utils.visualizer import ColorMode, Visualizer


class WheatEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

        # for visualization
        self._visresult = False
        self._id2anno = defaultdict(list)
        for anno in self._coco_api.dataset["annotations"]:
            ann = copy.deepcopy(anno)
            ann['bbox_mode'] = BoxMode.XYWH_ABS
            ann['category_id'] -= 1
            self._id2anno[ann['image_id']].append(ann)

        # score thresh
        if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
            self._score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        elif cfg.MODEL.META_ARCHITECTURE == "OneStageDetector":
            self._score_thresh = cfg.MODEL.FCOS.INFERENCE_TH_TEST
        else:
            self._logger.warning(f"score thresh is not implement for {cfg.MODEL.META_ARCHITECTURE}")
            self._score_thresh = 0.4

    def reset(self):
        self._predictions = []
        self._TTA_gts = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # self._id2img[input["image_id"]] = input["file_name"]

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])

                # draw prediction and anno at the same time
                if self._visresult:
                    self._vis_result(input["image_id"], input["file_name"], instances)

            # TODO: deal with the gt in inputs after augmentation
            if "instances" in input:
                instances = input["instances"].to(self._cpu_device)
                coco_instances = aug_gt_instances_to_coco_json(
                    instances, input["image_id"], input["height"], input["width"]
                )
                aug_gt = {
                    "image_id": input["image_id"],
                    "file_name": os.path.basename(input["file_name"]),
                    "height": input["height"],
                    "width": input["width"],
                    "instances": coco_instances,
                }
                self._TTA_gts.append(aug_gt)

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            aug_gts = comm.gather(self._TTA_gts, dst=0)
            aug_gts = list(itertools.chain(*aug_gts))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            aug_gts = self._TTA_gts

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        # aug_gts > 0 means use the aug label
        if len(aug_gts) > 0:
            tta_json_file = os.path.join(self._output_dir, "tta_dataset.json")
            aug_gt_convert_to_coco_json(aug_gts, output_file=tta_json_file)

            # update dataset
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_api = COCO(tta_json_file)

            self._id2anno.clear()
            for anno in self._coco_api.dataset["annotations"]:
                ann = copy.deepcopy(anno)
                ann['bbox_mode'] = BoxMode.XYWH_ABS
                ann['category_id'] -= 1
                self._id2anno[ann['image_id']].append(ann)

        # run evaluation
        self._results = OrderedDict()

        if "instances" in predictions[0]:
            # run coco evaluation
            self._eval_predictions(set(self._tasks), predictions)

            # self._results['bbox'] = {}
            # run wheat evaluation
            # oof_score = calculate_final_score(predictions, self._id2anno, score_threshold=self._score_thresh)
            # self._results['bbox'].update({"Score": oof_score})
            # self._logger.info(f"OOF score is {oof_score}")

            # calculate the best threshold
            metric = calculate_best_threshold(predictions, self._id2anno)
            # self._logger.info(f"best score is {best_score}, best threshold {best_threshold}")
            if 'bbox' not in self._results:
                self._results['bbox'] = {}
            self._results['bbox'].update(metric)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _vis_result(self, img_id, img_path, pred):
        from PIL import Image
        vis_output = None
        image = np.array(Image.open(img_path))
        ## Convert image from OpenCV BGR format to Matplotlib RGB format.
        #image = image[:, :, ::-1]
        visualizer = Visualizer(image, self._metadata)
        # draw prediction
        visualizer.draw_instance_predictions(pred)

        anno = {"annotations": self._id2anno[img_id] if len(self._id2anno[img_id]) > 0 else None}
        visualizer.draw_dataset_dict(anno)

        vis_output = visualizer.get_output()
        output_path = os.path.join(self._output_dir, img_path.split('/')[-1])
        vis_output.save(output_path)

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def aug_gt_instances_to_coco_json(instances, img_id, output_height, output_width):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    # 1. scale box to output size
    img_size = instances.image_size   # h, w
    scale_x, scale_y = (output_width / img_size[1], output_height / img_size[0])
    results = Instances((output_height, output_width), **instances.get_fields())

    output_boxes = instances.gt_boxes
    output_boxes.scale(scale_x, scale_y)    # xyxy
    output_boxes.clip(results.image_size)

    instances = results[output_boxes.nonempty()]

    # 2. convert to coco
    boxes = instances.gt_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    classes = instances.gt_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k] + 1,
            "bbox": boxes[k],
            "area": boxes[k][2] * boxes[k][3],
            "iscrowd": 0,
        }
        results.append(result)
    return results


def aug_gt_convert_to_coco_json(aug_gts, output_file):
    dataset = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    dataset['categories'].append({
        'id': 1,
        'name': "wheat",
        'supercategory': "wheat",
        'skeleton': []
    })

    box_id = 0
    for gt in aug_gts:
        dataset['images'].append({
            'coco_url': '',
            'date_captured': '',
            'file_name': gt['file_name'],
            'flickr_url': '',
            'id': gt['image_id'],
            'license': 0,
            'width': gt['width'],
            'height': gt['height'],
        })
        for anno in gt['instances']:
            anno['id'] = box_id
            dataset['annotations'].append(anno)
            box_id += 1

    # logger.info(f"Writing COCO format annotations at '{output_file}' ...")
    with PathManager.open(output_file, "w") as f:
        json.dump(dataset, f)


"""
MAP for Kaggle Wheat!!!!
from https://www.kaggle.com/pestipeti/competition-metric-details-script
"""


@jit(nopython=True)
def calculate_iou(gt, pr, form='pascal_voc') -> float:
    """Calculates the Intersection over Union.

    Args:
        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        (float) Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        gt = gt.copy()
        pr = pr.copy()

        gt[2] = gt[0] + gt[2]
        gt[3] = gt[1] + gt[3]
        pr[2] = pr[0] + pr[2]
        pr[3] = pr[1] + pr[3]

    # Calculate overlap area
    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0:
        return 0.0

    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0:
        return 0.0

    overlap_area = dx * dy

    # Calculate union area
    union_area = (
        (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
        (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
        overlap_area
    )

    return overlap_area / union_area


@jit(nopython=True)
def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        pred: (List[Union[int, float]]) Coordinates of the predicted box
        pred_idx: (int) Index of the current predicted box
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (int) Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx in range(len(gts)):

        if gts[gt_idx][0] < 0:
            # Already matched GT-box
            continue

        iou = -1 if ious is None else ious[gt_idx][pred_idx]

        if iou < 0:
            iou = calculate_iou(gts[gt_idx], pred, form=form)

            if ious is not None:
                ious[gt_idx][pred_idx] = iou

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx


@jit(nopython=True)
def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
    """Calculates precision for GT - prediction pairs at one threshold.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        threshold: (float) Threshold
        form: (str) Format of the coordinates
        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

    Return:
        (float) Precision
    """
    n = len(preds)
    tp = 0
    fp = 0

    # for pred_idx, pred in enumerate(preds_sorted):
    for pred_idx in range(n):
        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,
                                            threshold=threshold, form=form, ious=ious)

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1
            # Remove the matched GT box
            gts[best_match_gt_idx] = -1

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fp += 1

    # False negative: indicates a gt box had no associated predicted box.
    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)


@jit(nopython=True)
def calculate_image_precision(gts, preds, thresholds=(0.5,), form='coco') -> float:
    """Calculates image precision.

    Args:
        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
               sorted by confidence value (descending)
        thresholds: (float) Different thresholds
        form: (str) Format of the coordinates

    Return:
        (float) Precision
    """
    n_threshold = len(thresholds)
    image_precision = 0.0

    ious = np.ones((len(gts), len(preds))) * -1
    # ious = None

    for threshold in thresholds:
        if len(gts) > 0:
            precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,
                                                         form=form, ious=ious)
        image_precision += precision_at_threshold / n_threshold

    return image_precision


def calculate_final_score(all_predictions, gts, score_threshold):
    final_scores = []
    iou_thresholds = np.array([x for x in np.arange(0.5, 0.76, 0.05)])
    for i in range(len(all_predictions)):
        image_id = all_predictions[i]["image_id"]
        pred_instances = all_predictions[i]["instances"].copy()
        scores = []
        bboxes = []
        for inst in pred_instances:
            scores.append(inst["score"])
            bboxes.append(inst["bbox"])

        gt_bboxes = []
        for gt in gts[image_id]:
            gt_bboxes.append(gt['bbox'])

        # gt_boxes = all_predictions[i]['gt_boxes'].copy()
        # pred_boxes = all_predictions[i]['pred_boxes'].copy()
        # scores = all_predictions[i]['scores'].copy()
        # image_id = all_predictions[i]['image_id']

        scores = np.array(scores)
        pred_boxes = np.array(bboxes, dtype=np.int)
        preds_sorted_idx = np.argsort(scores)[::-1]
        pred_boxes = pred_boxes[preds_sorted_idx]

        gt_bboxes = np.array(gt_bboxes, dtype=np.int)

        indexes = np.where(scores > score_threshold)
        pred_boxes = pred_boxes[indexes]
        scores = scores[indexes]

        if len(gt_bboxes) == 0 and len(pred_boxes) == 0:
            image_precision = 1.0
        elif len(gt_bboxes) == 0 or len(pred_boxes) == 0:
            image_precision = 0.0
        else:
            image_precision = calculate_image_precision(gt_bboxes, pred_boxes, thresholds=iou_thresholds, form='coco')

        final_scores.append(image_precision)

    return np.mean(final_scores)


def calculate_best_threshold(all_predictions, gts):
    metric = {}
    best_final_score, best_score_threshold = 0, 0
    count = 0
    for score_threshold in tqdm(np.arange(0.2, 0.8, 0.01), total=np.arange(0.2, 0.8, 0.01).shape[0]):
        final_score = calculate_final_score(all_predictions, gts, score_threshold)
        if final_score > best_final_score:
            best_final_score = final_score
            best_score_threshold = score_threshold
        if count % 5 == 0:
            metric[f"th@{score_threshold:.2f}"] = final_score
        count += 1

    metric["best_th"] = best_score_threshold
    metric["best_score"] = best_final_score
    return metric