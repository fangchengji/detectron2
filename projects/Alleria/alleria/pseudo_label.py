#!/usr/bin/env python3
# @Time    : 24/6/20 5:00 PM
# @Author  : fangcheng.ji
# @FileName: pseudo_label.py

import os
from glob import glob
from detectron2.structures import BoxMode, Instances
from detectron2.engine import DefaultTrainer, hooks
from detectron2.utils import comm
from fvcore.nn.precise_bn import get_bn_modules

from alleria.data.datasets import register_coco_instances, _get_coco_meta
from .data.loader import build_detection_train_loader


# INPUT = "/kaggle/input/global-wheat-detection/test"
# OUTPUT = "/kaggle/working"

# INPUT = "/data/fangcheng.ji/datasets/wheat/pseudo_test"
# OUTPUT = "/data/fangcheng.ji/datasets/wheat/pseudo_test_out"


# _PSEUDO_DATASETS = {
#     "wheat_coco_pseudo": (
#         INPUT,
#         OUTPUT + "/pseudo_label.json",
#     ),
# }
#
# def register_pseudo_datasets(root=""):
#    for key, (image_root, json_file) in _PSEUDO_DATASETS.items():
#        # Assume pre-defined datasets live in `../datasets`.
#
#        register_coco_instances(
#            key,
#            _get_coco_meta(),
#            os.path.join(root, json_file) if "://" not in json_file else json_file,
#            os.path.join(root, image_root),
#        )


def register_pseudo_datasets(image_root, json_file):
    register_coco_instances(
        "wheat_coco_pseudo",
        _get_coco_meta(),
        json_file=json_file,
        image_root=image_root,
    )


def pred_instances_to_coco_json(instances, img_path, img_id, instance_id, img_size, score_threshold):
    # 1. score threshold filter
    scores = instances.scores
    instances = instances[scores > score_threshold]

    num_instance = len(instances)
    if num_instance == 0:
        return None, None, instance_id

    # 2. scale box to image size
    pred_img_size = instances.image_size   # h, w
    scale_x, scale_y = (img_size[1] / pred_img_size[1], img_size[0] / pred_img_size[0])
    results = Instances(img_size, **instances.get_fields())

    pred_boxes = instances.pred_boxes
    pred_boxes.scale(scale_x, scale_y)    # xyxy
    pred_boxes.clip(img_size)
    results.pred_boxes = pred_boxes

    instances = results[pred_boxes.nonempty()]

    # 3. convert to coco
    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    classes = instances.pred_classes.tolist()

    annos = []
    for k in range(len(instances)):
        anno = {
            "image_id": img_id,
            "category_id": classes[k] + 1,
            "bbox": boxes[k],
            "area": boxes[k][2] * boxes[k][3],
            "iscrowd": 0,
            "id": instance_id + k,
        }
        annos.append(anno)

    instance_id += len(instances)
    img = {
        "coco_url": "",
        "date_captured": "",
        "file_name": os.path.basename(img_path),
        "flickr_url": "",
        "id": img_id,
        "license": 0,
        "width": img_size[1],
        "height": img_size[0],
    }
    return img, annos, instance_id


def set_pseudo_cfg(cfg, img_nums, output_dir):
    cfg.defrost()

    cfg.DATASETS.TRAIN = ("wheat_coco_pseudo",)
    # need to skip the evaluation hook
    cfg.DATASETS.TEST = ("wheat_coco_pseudo",)

    cfg.SOLVER.IMS_PER_BATCH = 4

    epoch = 10

    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.WARMUP_FACTOR = 0.00005
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.MAX_ITER = img_nums * epoch // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.STEPS = (int(0.66 * cfg.SOLVER.MAX_ITER), int(0.88 * cfg.SOLVER.MAX_ITER))

    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.OUTPUT_DIR = output_dir

    cfg.freeze()
    return cfg


class PseudoTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    # exclude evaluation and periodcheckpoint hook
    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        # def test_and_save_results():
        #     self._last_eval_results = self.test(self.cfg, self.model)
        #     return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
